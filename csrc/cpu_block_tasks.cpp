#include <torch/extension.h>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <vector>

void commit_cpu_block_tasks(
    const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& mappings,
    const std::vector<torch::Tensor>& new_geometry,
    const std::vector<torch::Tensor>& new_sh,
    const std::vector<torch::Tensor>& destinations,
    int64_t block_size) {
  TORCH_CHECK(block_size > 0 && tables.size() >= 2, "expected parameter tables and positive block size");
  const int64_t tasks = mappings.size();
  TORCH_CHECK(new_geometry.size() == tasks && new_sh.size() == tasks && destinations.size() == tasks,
              "task vectors must have matching lengths");
  int64_t max_columns = 0;
  const int64_t capacity = tables[0].size(0);
  for (const auto& table : tables) {
    TORCH_CHECK(table.device().is_cpu() && table.scalar_type() == torch::kFloat32 &&
                table.dim() == 2 && table.is_contiguous() && table.size(0) == capacity,
                "tables must be contiguous CPU float32 with the same capacity");
    max_columns = std::max(max_columns, table.size(1));
  }
  TORCH_CHECK(tables[0].size(1) == 11 && tables[1].size(1) == 48, "expected geometry then SH");
  std::unordered_set<int64_t> written_blocks;
  // Validate all ownership and address ranges before the first write.
  for (int64_t task = 0; task < tasks; ++task) {
    const auto& mapping = mappings[task];
    const auto& dest = destinations[task];
    TORCH_CHECK(mapping.device().is_cpu() && mapping.scalar_type() == torch::kInt64 &&
                mapping.dim() == 1 && mapping.is_contiguous(), "invalid row mapping");
    TORCH_CHECK(dest.device().is_cpu() && dest.scalar_type() == torch::kInt64 &&
                dest.dim() == 2 && dest.size(1) == 3 && dest.is_contiguous(),
                "destinations must be CPU int64 [blocks, (id, count, skip)]");
    for (int field = 0; field < 2; ++field) {
      const auto& values = field == 0 ? new_geometry[task] : new_sh[task];
      TORCH_CHECK(values.device().is_cpu() && values.scalar_type() == torch::kFloat32 &&
                  values.dim() == 2 && values.size(1) == tables[field].size(1) && values.is_contiguous(),
                  "invalid new parameter rows");
    }
    TORCH_CHECK(new_geometry[task].size(0) == new_sh[task].size(0), "new parameter row counts differ");
    const auto* ranges = dest.data_ptr<int64_t>();
    const auto* rows = mapping.data_ptr<int64_t>();
    std::unordered_set<int64_t> owned;
    int64_t cursor = 0;
    for (int64_t b = 0; b < dest.size(0); ++b) {
      const int64_t id = ranges[3 * b], count = ranges[3 * b + 1], skip = ranges[3 * b + 2];
      TORCH_CHECK(id >= 0 && id < capacity / block_size && count >= 0 && count <= block_size &&
                  skip >= 0 && skip <= count && cursor + count <= mapping.numel(), "invalid block destination");
      TORCH_CHECK(written_blocks.insert(id).second, "CPU block tasks overlap");
      owned.insert(id);
      for (int64_t row = 0; row < skip; ++row) {
        TORCH_CHECK(rows[cursor + row] == id * block_size + row, "skipped prefix must be unchanged");
      }
      cursor += count;
    }
    TORCH_CHECK(cursor == mapping.numel(), "mapping length does not match destination counts");
    for (int64_t row = 0; row < mapping.numel(); ++row) {
      const int64_t source = rows[row];
      if (source >= 0) {
        TORCH_CHECK(source < capacity && owned.count(source / block_size),
                    "source rows must belong to the same disjoint task");
      } else {
        TORCH_CHECK(source >= -new_geometry[task].size(0), "new row index out of range");
      }
    }
  }
  pybind11::gil_scoped_release release;
  #pragma omp parallel for schedule(dynamic, 1) if(tasks > 1)
  for (int64_t task = 0; task < tasks; ++task) {
    const auto* rows = mappings[task].data_ptr<int64_t>();
    const auto* ranges = destinations[task].data_ptr<int64_t>();
    const int64_t blocks = destinations[task].size(0);
    int64_t moved = 0;
    for (int64_t b = 0; b < blocks; ++b) moved += ranges[3 * b + 1] - ranges[3 * b + 2];
    std::vector<float> scratch(moved * max_columns);
    for (size_t field = 0; field < tables.size(); ++field) {
      const int64_t columns = tables[field].size(1);
      auto* target = tables[field].data_ptr<float>();
      const float* added = field == 0 ? new_geometry[task].data_ptr<float>() :
                           field == 1 ? new_sh[task].data_ptr<float>() : nullptr;
      int64_t cursor = 0, write = 0;
      for (int64_t b = 0; b < blocks; ++b) {
        const int64_t count = ranges[3 * b + 1], skip = ranges[3 * b + 2];
        for (int64_t row = skip; row < count; ++row, ++write) {
          const int64_t source = rows[cursor + row];
          float* output = scratch.data() + write * columns;
          if (source >= 0) std::memcpy(output, target + source * columns, columns * sizeof(float));
          else if (added) std::memcpy(output, added + (-source - 1) * columns, columns * sizeof(float));
          else std::fill_n(output, columns, 0.0f);
        }
        cursor += count;
      }
      write = 0;
      for (int64_t b = 0; b < blocks; ++b) {
        const int64_t start = ranges[3 * b] * block_size;
        const int64_t count = ranges[3 * b + 1], skip = ranges[3 * b + 2];
        if (count > skip) std::memcpy(target + (start + skip) * columns,
                                    scratch.data() + write * columns, (count - skip) * columns * sizeof(float));
        std::fill(target + (start + count) * columns, target + (start + block_size) * columns, 0.0f);
        write += count - skip;
      }
    }
  }
}
