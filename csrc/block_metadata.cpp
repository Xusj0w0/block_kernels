#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

std::vector<torch::Tensor> cpu_block_bounds(
    torch::Tensor points, torch::Tensor radii, torch::Tensor offsets) {
  TORCH_CHECK(points.device().is_cpu() && points.scalar_type() == torch::kFloat32 &&
              points.dim() == 2 && points.size(1) == 3, "points must be CPU float32 [N,3]");
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type() == torch::kInt64 &&
              offsets.dim() == 1 && offsets.is_contiguous() && offsets.numel() > 0,
              "offsets must be contiguous CPU int64 [B+1]");
  const bool has_radii = radii.numel() != 0;
  TORCH_CHECK(radii.device().is_cpu() && radii.scalar_type() == torch::kFloat32,
              "radii must be CPU float32");
  TORCH_CHECK(!has_radii || (radii.size(0) == points.size(0) &&
              (radii.dim() == 1 || (radii.dim() == 2 &&
                (radii.size(1) == 1 || radii.size(1) == 3)))), "radii must be [N], [N,1], or [N,3]");
  const int64_t blocks = offsets.numel() - 1;
  const int64_t* ranges = offsets.data_ptr<int64_t>();
  TORCH_CHECK(ranges[0] == 0 && ranges[blocks] == points.size(0), "offsets must cover all points");
  for (int64_t b = 0; b < blocks; ++b) {
    TORCH_CHECK(ranges[b] >= 0 && ranges[b] <= ranges[b + 1], "offsets must be nondecreasing");
  }
  auto minimum = torch::empty({blocks, 3}, points.options());
  auto maximum = torch::empty({blocks, 3}, points.options());
  const float* xyz = points.data_ptr<float>();
  const float* radius = has_radii ? radii.data_ptr<float>() : nullptr;
  const int64_t ps0 = points.stride(0), ps1 = points.stride(1);
  const int64_t rs0 = has_radii ? radii.stride(0) : 0;
  const int64_t rs1 = has_radii && radii.dim() == 2 && radii.size(1) == 3 ? radii.stride(1) : 0;
  float* lo = minimum.data_ptr<float>();
  float* hi = maximum.data_ptr<float>();
  pybind11::gil_scoped_release release;
  #pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < blocks; ++b) {
    float lower[3] = {INFINITY, INFINITY, INFINITY};
    float upper[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int64_t row = ranges[b]; row < ranges[b + 1]; ++row) {
      for (int c = 0; c < 3; ++c) {
        const float r = radius ? radius[row * rs0 + c * rs1] : 0.0f;
        const float low = xyz[row * ps0 + c * ps1] - r;
        const float high = xyz[row * ps0 + c * ps1] + r;
        if (std::isnan(low) || low < lower[c]) lower[c] = low;
        if (std::isnan(high) || high > upper[c]) upper[c] = high;
      }
    }
    for (int c = 0; c < 3; ++c) {
      lo[b * 3 + c] = ranges[b] == ranges[b + 1] ? 0.0f : lower[c];
      hi[b * 3 + c] = ranges[b] == ranges[b + 1] ? 0.0f : upper[c];
    }
  }
  return {minimum, maximum};
}
