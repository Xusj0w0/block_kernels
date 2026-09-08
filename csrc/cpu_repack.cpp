#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// Gather a permutation with holes without a second parameter/state allocation.
// The mapping is destination -> original source. Components are paths ending
// outside the dense destination prefix, or cycles needing one saved row.
std::vector<torch::Tensor> repack_cpu_blocks(
    const std::vector<torch::Tensor>& tables, torch::Tensor order,
    torch::Tensor counts, int64_t block_size) {
  TORCH_CHECK(block_size > 0 && tables.size() >= 2, "expected geometry and SH tables");
  TORCH_CHECK(order.device().is_cpu() && order.scalar_type() == torch::kInt64 &&
              order.dim() == 1 && order.is_contiguous(), "order must be contiguous CPU int64");
  TORCH_CHECK(counts.device().is_cpu() && counts.scalar_type() == torch::kInt64 &&
              counts.dim() == 1 && counts.is_contiguous(), "counts must be contiguous CPU int64");
  for (const auto& table : tables) {
    TORCH_CHECK(table.device().is_cpu() && table.scalar_type() == torch::kFloat32 &&
                table.dim() == 2 && table.is_contiguous(), "invalid CPU float32 table");
    TORCH_CHECK(table.size(0) == tables[0].size(0) && table.size(1) > 0,
                "table capacities must match");
  }
  TORCH_CHECK(tables[0].size(1) == 11 && tables[1].size(1) == 48, "expected geometry then SH");
  TORCH_CHECK(counts.numel() <= tables[0].size(0) / block_size, "counts exceed capacity");
  // Parallel table edits require disjoint storage.
  for (size_t a = 0; a < tables.size(); ++a) for (size_t b = 0; b < a; ++b) {
    auto begin_a = reinterpret_cast<uintptr_t>(tables[a].data_ptr());
    auto begin_b = reinterpret_cast<uintptr_t>(tables[b].data_ptr());
    TORCH_CHECK(begin_a + tables[a].nbytes() <= begin_b || begin_b + tables[b].nbytes() <= begin_a,
                "repack tables must not overlap");
  }
  const int64_t old_slots = counts.numel() * block_size, n = order.numel();
  const auto* old_counts = counts.data_ptr<int64_t>();
  const auto* source = order.data_ptr<int64_t>();
  int64_t total = 0;
  for (int64_t b = 0; b < counts.numel(); ++b) {
    TORCH_CHECK(old_counts[b] >= 0 && old_counts[b] <= block_size, "invalid block count");
    total += old_counts[b];
  }
  TORCH_CHECK(n == total, "order must contain every live row");
  auto has = [](const std::vector<uint64_t>& bits, int64_t i) {
    return (bits[i / 64] >> (i % 64)) & 1ULL;
  };
  auto mark = [](std::vector<uint64_t>& bits, int64_t i) { bits[i / 64] |= 1ULL << (i % 64); };
  std::vector<uint64_t> incoming((old_slots + 63) / 64, 0);
  // Validate the entire mapping before changing any table.
  for (int64_t i = 0; i < n; ++i) {
    const int64_t s = source[i];
    TORCH_CHECK(s >= 0 && s < old_slots && s % block_size < old_counts[s / block_size],
                "order selects an invalid or empty slot");
    TORCH_CHECK(!has(incoming, s), "order contains duplicate source rows");
    mark(incoming, s);
  }
  const int64_t blocks = (n + block_size - 1) / block_size;
  auto new_counts = torch::full({blocks}, block_size, torch::kInt64);
  if (blocks) new_counts.data_ptr<int64_t>()[blocks - 1] = n - (blocks - 1) * block_size;
  auto minimum = torch::empty({blocks, 3}, torch::kFloat32);
  auto maximum = torch::empty_like(minimum);
  // Allocate all scratch before mutation, including per-table visitation bits.
  std::vector<std::vector<uint64_t>> visited(tables.size(), std::vector<uint64_t>((n + 63) / 64, 0));
  std::vector<std::vector<float>> saved(tables.size());
  for (size_t t = 0; t < tables.size(); ++t) saved[t].resize(tables[t].size(1));
  pybind11::gil_scoped_release release;
  #pragma omp parallel for schedule(static)
  for (size_t t = 0; t < tables.size(); ++t) {
    float* data = tables[t].data_ptr<float>();
    const int64_t width = tables[t].size(1);
    const size_t bytes = width * sizeof(float);
    auto move_component = [&](int64_t start, bool cycle) {
      if (cycle) std::memcpy(saved[t].data(), data + start * width, bytes);
      int64_t dest = start;
      while (true) {
        const int64_t s = source[dest];
        mark(visited[t], dest);
        if (cycle && s == start) {
          std::memcpy(data + dest * width, saved[t].data(), bytes);
          break;
        }
        std::memcpy(data + dest * width, data + s * width, bytes);
        if (s >= n) break;
        dest = s;
      }
    };
    // Start at path heads, not interior destinations whose old value is needed.
    for (int64_t i = 0; i < n; ++i) if (!has(incoming, i)) move_component(i, false);
    for (int64_t i = 0; i < n; ++i) if (!has(visited[t], i)) move_component(i, true);
    std::fill(data + n * width, data + old_slots * width, 0.0f);
  }
  const float* geometry = tables[0].data_ptr<float>();
  #pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < blocks; ++b) {
    float* lo = minimum.data_ptr<float>() + 3 * b;
    float* hi = maximum.data_ptr<float>() + 3 * b;
    std::fill_n(lo, 3, std::numeric_limits<float>::infinity());
    std::fill_n(hi, 3, -std::numeric_limits<float>::infinity());
    for (int64_t i = b * block_size; i < std::min(n, (b + 1) * block_size); ++i) {
      const float* p = geometry + i * 11;
      float radius = 3.0f * std::max({std::exp(p[4]), std::exp(p[5]), std::exp(p[6])});
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], p[a] - radius);
        hi[a] = std::max(hi[a], p[a] + radius);
      }
    }
  }
  return {new_counts, minimum, maximum};
}
