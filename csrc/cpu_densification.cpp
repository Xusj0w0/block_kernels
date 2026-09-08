#include <torch/extension.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <vector>

void commit_cpu_block_tasks(const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
                           const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
                           const std::vector<torch::Tensor>&, int64_t);

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
torch::Tensor longs(const std::vector<int64_t>& values) {
  auto out = torch::empty({static_cast<int64_t>(values.size())}, torch::kInt64);
  if (!values.empty()) std::memcpy(out.data_ptr(), values.data(), values.size() * sizeof(int64_t));
  return out;
}
float max_scale(const float* p) {
  if (std::isnan(p[4]) || std::isnan(p[5]) || std::isnan(p[6])) return NAN;
  return std::max({std::exp(p[4]), std::exp(p[5]), std::exp(p[6])});
}
void check_tables(const std::vector<torch::Tensor>& tables, torch::Tensor counts, int64_t size) {
  TORCH_CHECK(size > 0 && tables.size() >= 2, "invalid block size or tables");
  TORCH_CHECK(counts.device().is_cpu() && counts.scalar_type() == torch::kInt64 && counts.dim() == 1 &&
              counts.is_contiguous(), "counts must be CPU int64");
  for (const auto& table : tables) {
    TORCH_CHECK(table.device().is_cpu() && table.scalar_type() == torch::kFloat32 && table.dim() == 2 &&
                table.is_contiguous() && table.size(0) == tables[0].size(0), "invalid fixed-capacity CPU table");
  }
  TORCH_CHECK(tables[0].size(1) == 11 && tables[1].size(1) == 48, "expected geometry and SH tables");
  TORCH_CHECK(counts.numel() <= tables[0].size(0) / size, "layout exceeds prealloc_capacity");
  const auto* data = counts.data_ptr<int64_t>();
  for (int64_t b = 0; b < counts.numel(); ++b) TORCH_CHECK(data[b] >= 0 && data[b] <= size, "invalid block count");
}
struct Selection { std::vector<int64_t> clone, split; };
struct Tasks {
  std::vector<torch::Tensor> mapping, geometry, sh, destinations;
  explicit Tasks(size_t n) : mapping(n), geometry(n), sh(n), destinations(n) {}
  void commit(const std::vector<torch::Tensor>& tables, int64_t size) {
    commit_cpu_block_tasks(tables, mapping, geometry, sh, destinations, size);
  }
};
void bounds(const float* geometry, int64_t count, float* lo, float* hi) {
  for (int axis = 0; axis < 3; ++axis) { lo[axis] = INFINITY; hi[axis] = -INFINITY; }
  for (int64_t row = 0; row < count; ++row) {
    const float* p = geometry + row * 11;
    float radius = 3.0f * max_scale(p);
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = std::min(lo[axis], p[axis] - radius);
      hi[axis] = std::max(hi[axis], p[axis] + radius);
    }
  }
  if (!count) for (int axis = 0; axis < 3; ++axis) lo[axis] = hi[axis] = 0.0f;
}
// The same 10-bit normalization and stable ordering as block_layout.morton_sort.
void sort_rows(std::vector<int64_t>& rows, const float* geometry, const float* added) {
  std::array<float, 3> lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
  auto point = [&](int64_t id) { return id >= 0 ? geometry + id * 11 : added + (-id - 1) * 11; };
  for (int64_t id : rows) for (int a = 0; a < 3; ++a) {
    lo[a] = std::min(lo[a], point(id)[a]); hi[a] = std::max(hi[a], point(id)[a]);
  }
  std::vector<std::pair<uint32_t, int64_t>> keyed;
  keyed.reserve(rows.size());
  for (int64_t id : rows) {
    uint32_t key = 0;
    for (int a = 0; a < 3; ++a) {
      float v = std::clamp((point(id)[a] - lo[a]) / (hi[a] - lo[a] + 1e-8f), 0.0f, 1.0f);
      uint32_t q = std::isfinite(v) ? static_cast<uint32_t>(v * 1023.0f) : 0;
      for (int bit = 0; bit < 10; ++bit) key |= ((q >> bit) & 1U) << (3 * bit + a);
    }
    keyed.emplace_back(key, id);
  }
  std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  for (size_t i = 0; i < rows.size(); ++i) rows[i] = keyed[i].second;
}
void prepare_task(Tasks& tasks, int64_t task, const std::vector<int64_t>& blocks,
                  const std::vector<int64_t>& dest, const int64_t* counts,
                  const std::vector<Selection>& selected, const std::vector<int64_t>& split_offsets,
                  const float* noise, int64_t total_split, const std::vector<torch::Tensor>& tables,
                  int64_t size, bool reorder) {
  const float* geo = tables[0].data_ptr<float>();
  const float* sh = tables[1].data_ptr<float>();
  int64_t new_rows = 0;
  for (int64_t b : blocks) new_rows += selected[b].clone.size() + 2 * selected[b].split.size();
  auto& new_geo = tasks.geometry[task];
  auto& new_sh = tasks.sh[task];
  new_geo = torch::empty({new_rows, 11}, torch::kFloat32);
  new_sh = torch::empty({new_rows, 48}, torch::kFloat32);
  float* added = new_geo.data_ptr<float>();
  float* added_sh = new_sh.data_ptr<float>();
  std::vector<int64_t> rows;
  int64_t next = 0;
  for (int64_t b : blocks) {
    const auto& split = selected[b].split;
    size_t split_cursor = 0;
    for (int64_t r = 0; r < counts[b]; ++r) {
      int64_t id = b * size + r;
      if (split_cursor < split.size() && split[split_cursor] == id) ++split_cursor;
      else rows.push_back(id);
    }
    for (int64_t id : selected[b].clone) {
      std::memcpy(added + next * 11, geo + id * 11, 11 * sizeof(float));
      std::memcpy(added_sh + next * 48, sh + id * 48, 48 * sizeof(float));
      rows.push_back(-++next);
    }
    for (int copy = 0; copy < 2; ++copy) for (size_t s = 0; s < split.size(); ++s) {
      int64_t id = split[s];
      const float* p = geo + id * 11;
      float* child = added + next * 11;
      std::memcpy(child, p, 11 * sizeof(float));
      std::memcpy(added_sh + next * 48, sh + id * 48, 48 * sizeof(float));
      float norm = std::sqrt(p[7]*p[7] + p[8]*p[8] + p[9]*p[9] + p[10]*p[10]);
      float w=p[7]/norm, x=p[8]/norm, y=p[9]/norm, z=p[10]/norm;
      float rotation[9] = {1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                           2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                           2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)};
      const float* random = noise + (copy * total_split + split_offsets[b] + s) * 3;
      float displacement[3];
      for (int a = 0; a < 3; ++a) {
        float scale = std::exp(p[4+a]);
        displacement[a] = random[a] * scale;
        child[4+a] = std::log(scale / 1.6f);
      }
      for (int a = 0; a < 3; ++a) child[a] += rotation[3*a]*displacement[0] +
          rotation[3*a+1]*displacement[1] + rotation[3*a+2]*displacement[2];
      rows.push_back(-++next);
    }
  }
  if (reorder) sort_rows(rows, geo, added);
  tasks.mapping[task] = longs(rows);
  auto ranges = torch::empty({static_cast<int64_t>(dest.size()), 3}, torch::kInt64);
  auto* p = ranges.data_ptr<int64_t>();
  for (size_t i = 0; i < dest.size(); ++i) {
    p[3*i] = dest[i];
    p[3*i+1] = rows.size() / dest.size() + (i < rows.size() % dest.size());
    p[3*i+2] = !reorder && selected[blocks[0]].split.empty() ? counts[blocks[0]] : 0;
  }
  tasks.destinations[task] = ranges;
}
}

// Plan fields: counts, overflow IDs, bitset, source IDs/offsets, destination
// IDs/offsets, final counts, expansion IDs/min/max, stage-one timing/counts.
std::vector<torch::Tensor> densify_cpu_stage1(
    const std::vector<torch::Tensor>& tables, torch::Tensor gradients, torch::Tensor counts,
    int64_t size, double threshold, double dense_scale) {
  auto started = Clock::now();
  check_tables(tables, counts, size);
  const int64_t blocks = counts.numel();
  threshold = static_cast<float>(threshold); dense_scale = static_cast<float>(dense_scale);
  TORCH_CHECK(gradients.device().is_cpu() && gradients.scalar_type() == torch::kFloat32 &&
              gradients.is_contiguous() && gradients.numel() == blocks * size, "invalid mean-gradient array");
  const auto* old_counts = counts.data_ptr<int64_t>();
  const float* geo = tables[0].data_ptr<float>();
  const float* grads = gradients.data_ptr<float>();
  std::vector<Selection> selected(blocks);
  std::vector<int64_t> after(old_counts, old_counts + blocks), overflow, simple;
  {
    pybind11::gil_scoped_release release;
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < blocks; ++b) for (int64_t r = 0; r < old_counts[b]; ++r) {
      int64_t id = b * size + r;
      if (!(std::abs(grads[id]) >= threshold)) continue;
      float scale = max_scale(geo + id * 11);
      if (scale <= dense_scale) selected[b].clone.push_back(id);
      else if (scale > dense_scale) selected[b].split.push_back(id);
    }
  }
  int64_t clones = 0, splits = 0;
  for (int64_t b = 0; b < blocks; ++b) {
    clones += selected[b].clone.size(); splits += selected[b].split.size();
    after[b] += selected[b].clone.size() + selected[b].split.size();
    if (after[b] > size) overflow.push_back(b);
    else if (after[b] != old_counts[b]) simple.push_back(b);
  }
  double selection_seconds = elapsed(started);
  const int64_t words = (size + 31) / 32;
  auto bitset = torch::zeros({static_cast<int64_t>(overflow.size()), words}, torch::kInt32);
  auto* bits = reinterpret_cast<uint32_t*>(bitset.data_ptr<int32_t>());
  for (size_t i = 0; i < overflow.size(); ++i) {
    int64_t b = overflow[i];
    for (const auto* ids : {&selected[b].clone, &selected[b].split}) for (int64_t id : *ids)
      bits[i*words+(id%size)/32] |= 1U << ((id%size)%32);
  }
  auto simple_start = Clock::now();
  std::vector<int64_t> split_offsets(blocks+1, 0);
  for (int64_t b = 0; b < blocks; ++b) split_offsets[b+1] = split_offsets[b] + (after[b] <= size ? selected[b].split.size() : 0);
  auto noise = torch::randn({2, split_offsets.back(), 3}, torch::kFloat32);
  Tasks tasks(simple.size());
  auto expansion_lo = torch::empty({static_cast<int64_t>(simple.size()), 3}, torch::kFloat32);
  auto expansion_hi = torch::empty_like(expansion_lo);
  {
    pybind11::gil_scoped_release release;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(simple.size()); ++i) {
      int64_t b = simple[i];
      prepare_task(tasks, i, {b}, {b}, old_counts, selected, split_offsets,
                   noise.data_ptr<float>(), split_offsets.back(), tables, size, false);
    }
  }
  auto bounds_start = Clock::now();
  {
    pybind11::gil_scoped_release release;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(simple.size()); ++i)
      bounds(tasks.geometry[i].data_ptr<float>(), tasks.geometry[i].size(0),
             expansion_lo.data_ptr<float>() + i*3, expansion_hi.data_ptr<float>() + i*3);
  }
  double bounds_seconds = elapsed(bounds_start);
  tasks.commit(tables, size);
  auto current_counts = counts.clone();
  for (int64_t b : simple) current_counts.data_ptr<int64_t>()[b] = after[b];
  auto timings = torch::tensor({selection_seconds, bounds_seconds, elapsed(simple_start)-bounds_seconds}, torch::kFloat64);
  return {current_counts, longs(overflow), bitset, longs({}), longs({0}), longs({}),
          longs({0}), longs(after), longs(simple), expansion_lo, expansion_hi, timings,
          longs({clones, splits, static_cast<int64_t>(simple.size()), static_cast<int64_t>(overflow.size()),
                 0})};
}

std::vector<torch::Tensor> densify_cpu_stage2(
    const std::vector<torch::Tensor>& tables, const std::vector<torch::Tensor>& plan,
    int64_t size, double dense_scale, double min_opacity, double world_scale) {
  TORCH_CHECK(plan.size() == 13, "invalid density plan");
  check_tables(tables, plan[0], size);
  check_tables(tables, plan[7], size);
  for (int field : {1, 3, 4, 5, 6}) TORCH_CHECK(plan[field].device().is_cpu() &&
      plan[field].scalar_type() == torch::kInt64 && plan[field].dim() == 1 && plan[field].is_contiguous(),
      "invalid cluster plan tensor");
  dense_scale = static_cast<float>(dense_scale);
  min_opacity = static_cast<float>(min_opacity); world_scale = static_cast<float>(world_scale);
  const int64_t blocks = plan[0].numel(), words = (size+31)/32;
  TORCH_CHECK(plan[2].device().is_cpu() && plan[2].scalar_type() == torch::kInt32 &&
              plan[2].is_contiguous() && plan[2].dim() == 2 && plan[2].size(0) == plan[1].numel() &&
              plan[2].size(1) == words, "invalid overflow bitset");
  const auto* counts = plan[0].data_ptr<int64_t>();
  const auto* overflow = plan[1].data_ptr<int64_t>();
  const auto* bits = reinterpret_cast<const uint32_t*>(plan[2].data_ptr<int32_t>());
  const auto* source = plan[3].data_ptr<int64_t>();
  const auto* source_offsets = plan[4].data_ptr<int64_t>();
  const auto* dest = plan[5].data_ptr<int64_t>();
  const auto* dest_offsets = plan[6].data_ptr<int64_t>();
  const int64_t groups = plan[4].numel()-1;
  TORCH_CHECK(groups >= 0 && plan[6].numel() == groups+1 && source_offsets[0] == 0 && dest_offsets[0] == 0 &&
              source_offsets[groups] == plan[3].numel() && dest_offsets[groups] == plan[5].numel(),
              "invalid cluster offsets");
  std::vector<bool> is_overflow(blocks, false), used_source(blocks, false), used_dest(plan[7].numel(), false);
  std::vector<int64_t> added_counts(blocks, 0);
  for (int64_t i = 0; i < plan[1].numel(); ++i) {
    int64_t b = overflow[i];
    TORCH_CHECK(b >= 0 && b < blocks && !is_overflow[b], "invalid or duplicate overflow block");
    is_overflow[b] = true;
    for (int64_t r = 0; r < words*32; ++r) if (bits[i*words+r/32] & (1U << (r%32))) {
      TORCH_CHECK(r < counts[b], "overflow bitset selects an empty slot");
      ++added_counts[b];
    }
    TORCH_CHECK(counts[b] + added_counts[b] > size, "overflow block does not overflow");
  }
  for (int64_t i = 0; i < groups; ++i) {
    TORCH_CHECK(source_offsets[i+1] > source_offsets[i] && source_offsets[i+1] <= plan[3].numel() &&
                dest_offsets[i+1] > dest_offsets[i] && dest_offsets[i+1] <= plan[5].numel(), "invalid cluster range");
    std::unordered_set<int64_t> owned;
    int64_t points = 0, output_points = 0;
    for (int64_t j = dest_offsets[i]; j < dest_offsets[i+1]; ++j) {
      int64_t b = dest[j];
      TORCH_CHECK(b >= 0 && b < plan[7].numel() && !used_dest[b], "overlapping cluster destinations");
      used_dest[b] = true; owned.insert(b); output_points += plan[7].data_ptr<int64_t>()[b];
    }
    for (int64_t j = source_offsets[i]; j < source_offsets[i+1]; ++j) {
      int64_t b = source[j];
      TORCH_CHECK(b >= 0 && b < blocks && !used_source[b] && owned.count(b),
                  "cluster sources must be disjoint and included in destinations");
      used_source[b] = true; points += counts[b] + added_counts[b];
    }
    for (int64_t b : owned) TORCH_CHECK(b >= blocks || used_source[b] || counts[b] == 0,
                                      "additional destinations must be empty");
    TORCH_CHECK(points == output_points, "cluster point counts do not match");
    int64_t k = dest_offsets[i+1] - dest_offsets[i];
    for (int64_t j = 0; j < k; ++j) TORCH_CHECK(
        plan[7].data_ptr<int64_t>()[dest[dest_offsets[i]+j]] == points/k + (j < points%k),
        "cluster destinations must use balanced counts");
  }
  for (int64_t b = 0; b < blocks; ++b) TORCH_CHECK(!is_overflow[b] || used_source[b], "overflow block missing from clusters");
  auto repartition_start = Clock::now();
  std::vector<Selection> selected(blocks);
  const float* geo = tables[0].data_ptr<float>();
  for (int64_t i = 0; i < plan[1].numel(); ++i) {
    int64_t b = overflow[i];
    for (int64_t row = 0; row < counts[b]; ++row) if (bits[i*words+row/32] & (1U << (row%32))) {
      int64_t id = b*size+row;
      if (max_scale(geo+id*11) <= dense_scale) selected[b].clone.push_back(id);
      else selected[b].split.push_back(id);
    }
  }
  std::vector<int64_t> split_offsets(blocks+1, 0);
  for (int64_t b = 0; b < blocks; ++b) split_offsets[b+1] = split_offsets[b] + selected[b].split.size();
  auto noise = torch::randn({2, split_offsets.back(), 3}, torch::kFloat32);
  Tasks tasks(groups);
  {
    pybind11::gil_scoped_release release;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < groups; ++i) {
      std::vector<int64_t> members(source+source_offsets[i], source+source_offsets[i+1]);
      std::vector<int64_t> targets(dest+dest_offsets[i], dest+dest_offsets[i+1]);
      prepare_task(tasks, i, members, targets, counts, selected, split_offsets,
                   noise.data_ptr<float>(), split_offsets.back(), tables, size, true);
    }
  }
  tasks.commit(tables, size);
  double repartition_seconds = elapsed(repartition_start);
  auto metadata_start = Clock::now();
  auto current = plan[7].clone();
  auto* final_counts = current.data_ptr<int64_t>();
  auto minimum = torch::empty({plan[5].numel(), 3}, torch::kFloat32);
  auto maximum = torch::empty_like(minimum);
  {
    pybind11::gil_scoped_release release;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < plan[5].numel(); ++i)
      bounds(geo+dest[i]*size*11, final_counts[dest[i]], minimum.data_ptr<float>()+i*3, maximum.data_ptr<float>()+i*3);
  }
  double metadata_seconds = elapsed(metadata_start);
  auto prune_start = Clock::now();
  int64_t pruned = 0;
  {
    pybind11::gil_scoped_release release;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:pruned)
    for (int64_t b = 0; b < current.numel(); ++b) {
      const int64_t start = b*size, n = final_counts[b];
      int64_t kept = 0;
      for (int64_t row = 0; row < n; ++row) {
        const float* p = geo+(start+row)*11;
        bool remove = 1.0f/(1.0f+std::exp(-p[3])) < min_opacity;
        if (world_scale > 0) remove |= max_scale(p) > world_scale;
        if (remove) continue;
        if (kept != row) for (const auto& table : tables) {
          int64_t width = table.size(1);
          float* data = table.data_ptr<float>();
          std::memmove(data+(start+kept)*width, data+(start+row)*width, width*sizeof(float));
        }
        ++kept;
      }
      if (kept != n) for (const auto& table : tables) {
        int64_t width = table.size(1); float* data = table.data_ptr<float>();
        std::fill(data+(start+kept)*width, data+(start+size)*width, 0.0f);
      }
      final_counts[b] = kept; pruned += n-kept;
    }
  }
  return {current, minimum, maximum,
          torch::tensor({repartition_seconds, metadata_seconds, elapsed(prune_start)}, torch::kFloat64), longs({pruned})};
}
