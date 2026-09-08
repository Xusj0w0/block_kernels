#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include <cstdint>

namespace {

__device__ float minimum(float a, float b) {
  return isnan(a) ? a : (isnan(b) ? b : fminf(a, b));
}

__device__ float maximum(float a, float b) {
  return isnan(a) ? a : (isnan(b) ? b : fmaxf(a, b));
}

// Each CUDA block reduces one finished Gaussian block from its existing DMA buffer.
__global__ void update_metadata_kernel(
    const float* geometry, const int64_t* ranges,
    float* aabb_min, float* aabb_max, float* centers, float* radii) {
  const int64_t block = ranges[blockIdx.x * 3];
  const int64_t start = ranges[blockIdx.x * 3 + 1];
  const int64_t count = ranges[blockIdx.x * 3 + 2];
  float low[3] = {CUDART_INF_F, CUDART_INF_F, CUDART_INF_F};
  float high[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
  for (int64_t row = threadIdx.x; row < count; row += blockDim.x) {
    const float* point = geometry + (start + row) * 11;
    const float radius = 3.0f * maximum(maximum(expf(point[4]), expf(point[5])), expf(point[6]));
    for (int axis = 0; axis < 3; ++axis) {
      low[axis] = minimum(low[axis], point[axis] - radius);
      high[axis] = maximum(high[axis], point[axis] + radius);
    }
  }
  __shared__ float values[6][256];
  for (int axis = 0; axis < 3; ++axis) {
    values[axis][threadIdx.x] = low[axis];
    values[axis + 3][threadIdx.x] = high[axis];
  }
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      for (int axis = 0; axis < 3; ++axis) {
        values[axis][threadIdx.x] = minimum(values[axis][threadIdx.x], values[axis][threadIdx.x + stride]);
        values[axis + 3][threadIdx.x] = maximum(values[axis + 3][threadIdx.x], values[axis + 3][threadIdx.x + stride]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < 3) {
    const int axis = threadIdx.x;
    const int64_t output = block * 3 + axis;
    const float lo = count ? values[axis][0] : 0.0f;
    const float hi = count ? values[axis + 3][0] : 0.0f;
    aabb_min[output] = lo;
    aabb_max[output] = hi;
    centers[output] = (lo + hi) * 0.5f;
    radii[output] = count ? maximum((hi - lo) * 0.8660254037844386f, FLT_EPSILON) : 0.0f;
  }
}

}  // namespace

void update_metadata_from_geometry(
    torch::Tensor geometry, torch::Tensor ranges,
    torch::Tensor aabb_min, torch::Tensor aabb_max,
    torch::Tensor centers, torch::Tensor radii, uintptr_t stream_ptr) {
  TORCH_CHECK(geometry.is_cuda() && geometry.scalar_type() == torch::kFloat32 &&
              geometry.is_contiguous() && geometry.dim() == 2 && geometry.size(1) == 11,
              "geometry must be contiguous CUDA float32 [N, 11]");
  TORCH_CHECK(ranges.device() == geometry.device() && ranges.scalar_type() == torch::kInt64 &&
              ranges.is_contiguous() && ranges.dim() == 2 && ranges.size(1) == 3,
              "ranges must be contiguous CUDA int64 [K, 3] on the geometry device");
  for (const auto& output : {aabb_min, aabb_max, centers, radii}) {
    TORCH_CHECK(output.device() == geometry.device() && output.scalar_type() == torch::kFloat32 &&
                output.is_contiguous() && output.dim() == 2 && output.size(1) == 3 &&
                output.size(0) == aabb_min.size(0),
                "metadata must be contiguous CUDA float32 [B, 3] on the geometry device");
  }
  if (ranges.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(geometry.device());
  update_metadata_kernel<<<ranges.size(0), 256, 0, reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
      geometry.data_ptr<float>(), ranges.data_ptr<int64_t>(),
      aabb_min.data_ptr<float>(), aabb_max.data_ptr<float>(),
      centers.data_ptr<float>(), radii.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
