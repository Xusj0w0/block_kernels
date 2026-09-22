#include <torch/extension.h>
#include <cstdint>

void bind_lod_initialize(pybind11::module_&);

void copy_h2d_ranges(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);
void copy_d2h_ranges(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);
void copy_d2d_ranges(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);
void mark_compact_mask(torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor,
                       int64_t, int64_t, uintptr_t);
void fixed_grid_accumulate_d2h_blocks(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);
void accumulate_d2h_rows(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);
std::vector<torch::Tensor> cpu_block_bounds(torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor remap_cpu_rows(torch::Tensor, torch::Tensor);
torch::Tensor expand_cpu_ranges(torch::Tensor, torch::Tensor, bool);
torch::Tensor concat_cpu_rows(const std::vector<torch::Tensor>&);
std::vector<torch::Tensor> repack_cpu_blocks(const std::vector<torch::Tensor>&,
                                          torch::Tensor, torch::Tensor, int64_t);
void commit_cpu_block_tasks(const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
                            const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
                            const std::vector<torch::Tensor>&, int64_t);
std::vector<torch::Tensor> densify_cpu_stage1(const std::vector<torch::Tensor>&, torch::Tensor,
    torch::Tensor, int64_t, double, double);
std::vector<torch::Tensor> densify_cpu_stage2(const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&, int64_t, double, double, double);
void update_metadata_from_geometry(torch::Tensor, torch::Tensor, torch::Tensor,
                                   torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  bind_lod_initialize(module);
  module.def("copy_h2d_ranges", &copy_h2d_ranges);
  module.def("copy_d2h_ranges", &copy_d2h_ranges);
  module.def("copy_d2d_ranges", &copy_d2d_ranges);
  module.def("mark_compact_mask", &mark_compact_mask);
  module.def("fixed_grid_accumulate_d2h_blocks", &fixed_grid_accumulate_d2h_blocks);
  module.def("accumulate_d2h_rows", &accumulate_d2h_rows);
  module.def("cpu_block_bounds", &cpu_block_bounds);
  module.def("remap_cpu_rows", &remap_cpu_rows);
  module.def("expand_cpu_ranges", &expand_cpu_ranges,
             pybind11::arg("starts"), pybind11::arg("counts"), pybind11::arg("int32_output") = false);
  module.def("concat_cpu_rows", &concat_cpu_rows);
  module.def("repack_cpu_blocks", &repack_cpu_blocks);
  module.def("commit_cpu_block_tasks", &commit_cpu_block_tasks);
  module.def("densify_cpu_stage1", &densify_cpu_stage1);
  module.def("densify_cpu_stage2", &densify_cpu_stage2);
  module.def("update_metadata_from_geometry", &update_metadata_from_geometry);
}
