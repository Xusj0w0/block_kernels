#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

void check_range_inputs(
    const torch::Tensor& input,
    const torch::Tensor& output,
    const torch::Tensor& source_starts,
    const torch::Tensor& destination_starts,
    const torch::Tensor& counts) {
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2, "tensors must be two-dimensional");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32,
              "only float32 tensors are supported");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(source_starts.device().is_cpu() && destination_starts.device().is_cpu() && counts.device().is_cpu(),
              "range descriptors must be CPU tensors");
  TORCH_CHECK(source_starts.scalar_type() == torch::kInt64 &&
              destination_starts.scalar_type() == torch::kInt64 &&
              counts.scalar_type() == torch::kInt64,
              "range descriptors must be int64");
  TORCH_CHECK(source_starts.numel() == destination_starts.numel() &&
              source_starts.numel() == counts.numel(),
              "range descriptors must have equal lengths");
}

void check_cuda_stream(torch::Tensor tensor, uintptr_t stream) {
  TORCH_CHECK(tensor.is_cuda(), "device tensor must be CUDA");
  // stream == 0 is CUDA's legacy default stream and is valid.
}

}  // namespace

void copy_h2d_ranges(
    torch::Tensor output,
    torch::Tensor host,
    torch::Tensor source_starts,
    torch::Tensor destination_starts,
    torch::Tensor counts,
    uintptr_t stream_ptr) {
  check_range_inputs(host, output, source_starts, destination_starts, counts);
  check_cuda_stream(output, stream_ptr);
  TORCH_CHECK(!host.is_cuda() && host.is_pinned(), "host tensor must be pinned CPU memory");
  TORCH_CHECK(host.size(1) == output.size(1), "column count mismatch");
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const auto* source = source_starts.data_ptr<int64_t>();
  const auto* destination = destination_starts.data_ptr<int64_t>();
  const auto* length = counts.data_ptr<int64_t>();
  const int64_t columns = host.size(1);
  for (int64_t i = 0; i < counts.numel(); ++i) {
    TORCH_CHECK(source[i] >= 0 && destination[i] >= 0 && length[i] >= 0,
                "range values must be non-negative");
    TORCH_CHECK(source[i] + length[i] <= host.size(0) &&
                destination[i] + length[i] <= output.size(0),
                "range exceeds tensor bounds");
    C10_CUDA_CHECK(cudaMemcpyAsync(
        output.data_ptr<float>() + destination[i] * columns,
        host.data_ptr<float>() + source[i] * columns,
        static_cast<size_t>(length[i] * columns * sizeof(float)),
        cudaMemcpyHostToDevice,
        stream));
  }
}

void copy_d2h_ranges(
    torch::Tensor host,
    torch::Tensor input,
    torch::Tensor source_starts,
    torch::Tensor destination_starts,
    torch::Tensor counts,
    uintptr_t stream_ptr) {
  check_range_inputs(input, host, source_starts, destination_starts, counts);
  check_cuda_stream(input, stream_ptr);
  TORCH_CHECK(!host.is_cuda() && host.is_pinned(), "host tensor must be pinned CPU memory");
  TORCH_CHECK(host.size(1) == input.size(1), "column count mismatch");
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const auto* source = source_starts.data_ptr<int64_t>();
  const auto* destination = destination_starts.data_ptr<int64_t>();
  const auto* length = counts.data_ptr<int64_t>();
  const int64_t columns = host.size(1);
  for (int64_t i = 0; i < counts.numel(); ++i) {
    TORCH_CHECK(source[i] >= 0 && destination[i] >= 0 && length[i] >= 0,
                "range values must be non-negative");
    TORCH_CHECK(source[i] + length[i] <= input.size(0) &&
                destination[i] + length[i] <= host.size(0),
                "range exceeds tensor bounds");
    C10_CUDA_CHECK(cudaMemcpyAsync(
        host.data_ptr<float>() + destination[i] * columns,
        input.data_ptr<float>() + source[i] * columns,
        static_cast<size_t>(length[i] * columns * sizeof(float)),
        cudaMemcpyDeviceToHost,
        stream));
  }
}

void copy_d2d_ranges(
    torch::Tensor output,
    torch::Tensor input,
    torch::Tensor source_starts,
    torch::Tensor destination_starts,
    torch::Tensor counts,
    uintptr_t stream_ptr) {
  check_range_inputs(input, output, source_starts, destination_starts, counts);
  check_cuda_stream(output, stream_ptr);
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.size(1) == output.size(1), "column count mismatch");
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const auto* source = source_starts.data_ptr<int64_t>();
  const auto* destination = destination_starts.data_ptr<int64_t>();
  const auto* length = counts.data_ptr<int64_t>();
  const int64_t columns = input.size(1);
  for (int64_t i = 0; i < counts.numel(); ++i) {
    TORCH_CHECK(source[i] >= 0 && destination[i] >= 0 && length[i] >= 0,
                "range values must be non-negative");
    TORCH_CHECK(source[i] + length[i] <= input.size(0) &&
                destination[i] + length[i] <= output.size(0),
                "range exceeds tensor bounds");
    C10_CUDA_CHECK(cudaMemcpyAsync(
        output.data_ptr<float>() + destination[i] * columns,
        input.data_ptr<float>() + source[i] * columns,
        static_cast<size_t>(length[i] * columns * sizeof(float)),
        cudaMemcpyDeviceToDevice,
        stream));
  }
}

__global__ void mark_compact_mask_kernel(
    const int32_t* radii,
    const int64_t* block_ids,
    const int64_t* compact_starts,
    const int64_t* block_offsets,
    const int64_t* block_word_offsets,
    int64_t num_visible_blocks,
    uint32_t* mask_words) {
  for (int64_t block_rank = blockIdx.x;
       block_rank < num_visible_blocks;
       block_rank += gridDim.x) {
    const int64_t compact_start = compact_starts[block_rank];
    const int64_t block = block_ids[block_rank];
    const int64_t count = block_offsets[block + 1] - block_offsets[block];
    const int64_t word_start = block_word_offsets[block];
    for (int64_t row = threadIdx.x;
         row < count;
         row += blockDim.x) {
      if (radii[compact_start + row] > 0) {
        atomicOr(mask_words + word_start + (row >> 5),
                 1U << (row & 31));
      }
    }
  }
}

void mark_compact_mask(
    torch::Tensor radii,
    torch::Tensor block_ids,
    torch::Tensor compact_starts,
    torch::Tensor block_offsets,
    torch::Tensor block_word_offsets,
    torch::Tensor device_mask,
    torch::Tensor host_mask,
    int64_t finish_word_begin,
    int64_t finish_word_end,
    uintptr_t stream_ptr) {
  TORCH_CHECK(radii.is_cuda() && radii.scalar_type() == torch::kInt32 &&
              radii.is_contiguous(), "radii must be contiguous CUDA int32");
  TORCH_CHECK(block_ids.is_cuda() && block_ids.scalar_type() == torch::kInt64 &&
              block_ids.is_contiguous(), "block_ids must be contiguous CUDA int64");
  TORCH_CHECK(compact_starts.is_cuda() &&
              compact_starts.scalar_type() == torch::kInt64 &&
              compact_starts.is_contiguous() &&
              compact_starts.numel() == block_ids.numel(),
              "compact_starts must match block_ids");
  TORCH_CHECK(block_offsets.is_cuda() &&
              block_offsets.scalar_type() == torch::kInt64 &&
              block_offsets.is_contiguous() &&
              block_offsets.numel() == block_word_offsets.numel() + 1,
              "block_offsets must contain one entry per block plus a sentinel");
  TORCH_CHECK(block_word_offsets.is_cuda() &&
              block_word_offsets.scalar_type() == torch::kInt64 &&
              block_word_offsets.is_contiguous(),
              "block_word_offsets must be contiguous CUDA int64");
  TORCH_CHECK(device_mask.is_cuda() &&
              device_mask.scalar_type() == torch::kInt32 &&
              device_mask.is_contiguous(),
              "device_mask must be contiguous CUDA int32");
  TORCH_CHECK(!host_mask.is_cuda() && host_mask.is_pinned() &&
              host_mask.scalar_type() == torch::kInt32 &&
              host_mask.is_contiguous(),
              "host_mask must be contiguous pinned CPU int32");
  TORCH_CHECK(device_mask.numel() == host_mask.numel(),
              "device and host compact masks must have equal sizes");
  TORCH_CHECK(finish_word_begin >= 0 &&
              finish_word_begin <= finish_word_end &&
              finish_word_end <= device_mask.numel(),
              "finish word range is outside compact mask");

  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (block_ids.numel() > 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        block_ids.numel() > 65535 ? 65535 : block_ids.numel());
    mark_compact_mask_kernel<<<blocks, threads, 0, stream>>>(
        radii.data_ptr<int32_t>(), block_ids.data_ptr<int64_t>(),
        compact_starts.data_ptr<int64_t>(),
        block_offsets.data_ptr<int64_t>(),
        block_word_offsets.data_ptr<int64_t>(), block_ids.numel(),
        reinterpret_cast<uint32_t*>(device_mask.data_ptr<int32_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t finish_words = finish_word_end - finish_word_begin;
  if (finish_words > 0) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        host_mask.data_ptr<int32_t>() + finish_word_begin,
        device_mask.data_ptr<int32_t>() + finish_word_begin,
        static_cast<size_t>(finish_words) * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream));
  }
}

// ---------------------------------------------------------------------------
// Fixed-grid mapped-host accumulation used by the hybrid block backend.
//
// The original CLM kernels use one indexed CUDA launch for a selected set of
// rows.  BlockGS keeps the same launch/dataflow idea but supplies one
// descriptor per complete block (source row, destination row, row count).
// Each CUDA block handles one descriptor, so no block is split into separate
// host API copies and all 59 packed columns are transferred by the same
// kernel.  Descriptor tensors are CUDA int64 tensors, matching CLM's index
// representation and avoiding a CPU-side loop over ranges.
// ---------------------------------------------------------------------------

void check_fixed_grid_inputs(
    const torch::Tensor& input,
    const torch::Tensor& output,
    const torch::Tensor& source_starts,
    const torch::Tensor& destination_starts,
    const torch::Tensor& counts) {
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2, "tensors must be two-dimensional");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32,
              "only float32 tensors are supported");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(source_starts.is_cuda() && destination_starts.is_cuda() && counts.is_cuda(),
              "block descriptors must be CUDA tensors");
  TORCH_CHECK(source_starts.scalar_type() == torch::kInt64 &&
              destination_starts.scalar_type() == torch::kInt64 &&
              counts.scalar_type() == torch::kInt64,
              "block descriptors must be int64");
  TORCH_CHECK(source_starts.is_contiguous() && destination_starts.is_contiguous() && counts.is_contiguous(),
              "block descriptors must be contiguous");
  TORCH_CHECK(source_starts.numel() == destination_starts.numel() &&
              source_starts.numel() == counts.numel(),
              "block descriptors must have equal lengths");
  TORCH_CHECK(input.size(1) == output.size(1), "column count mismatch");
}

__global__ void fixed_grid_accumulate_blocks_kernel(
    const float* input,
    float* output,
    const int64_t* source_starts,
    const int64_t* destination_starts,
    const int64_t* counts,
    int64_t num_blocks,
    int64_t columns) {
  for (int64_t block_id = blockIdx.x;
       block_id < num_blocks;
       block_id += gridDim.x) {
    int64_t elements = counts[block_id] * columns;
    int64_t source = source_starts[block_id] * columns;
    int64_t destination = destination_starts[block_id] * columns;
    for (int64_t i = threadIdx.x; i < elements; i += blockDim.x) {
      output[destination + i] += input[source + i];
    }
  }
}

void fixed_grid_accumulate_d2h_blocks(
    torch::Tensor host,
    torch::Tensor input,
    torch::Tensor source_starts,
    torch::Tensor destination_starts,
    torch::Tensor counts,
    uintptr_t stream_ptr) {
  check_fixed_grid_inputs(input, host, source_starts, destination_starts, counts);
  TORCH_CHECK(!host.is_cuda() && host.is_pinned(), "host tensor must be pinned CPU memory");
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (counts.numel() == 0) return;
  constexpr unsigned int host_grid_size = 32;
  fixed_grid_accumulate_blocks_kernel<<<host_grid_size, 256, 0, stream>>>(
      input.data_ptr<float>(), host.data_ptr<float>(),
      source_starts.data_ptr<int64_t>(), destination_starts.data_ptr<int64_t>(),
      counts.data_ptr<int64_t>(), counts.numel(), input.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
