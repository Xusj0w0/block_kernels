#include <torch/extension.h>
#include <cstdint>

void copy_h2d_ranges(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);
void copy_d2h_ranges(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);
void copy_d2d_ranges(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);
void mark_compact_mask(torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor,
                       int64_t, int64_t, uintptr_t);
void fixed_grid_accumulate_d2h_blocks(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, uintptr_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("copy_h2d_ranges", &copy_h2d_ranges);
  module.def("copy_d2h_ranges", &copy_d2h_ranges);
  module.def("copy_d2d_ranges", &copy_d2d_ranges);
  module.def("mark_compact_mask", &mark_compact_mask);
  module.def("fixed_grid_accumulate_d2h_blocks", &fixed_grid_accumulate_d2h_blocks);
}
