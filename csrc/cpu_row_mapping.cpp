#include <torch/extension.h>
#include <cstring>
#include <limits>
#include <vector>

torch::Tensor concat_cpu_rows(const std::vector<torch::Tensor>& inputs) {
  TORCH_CHECK(!inputs.empty(), "cannot concatenate an empty tensor list");
  const auto& first = inputs.front();
  if (!first.device().is_cpu() || (first.dim() != 1 && first.dim() != 2) ||
      (first.scalar_type() != torch::kFloat32 && first.scalar_type() != torch::kInt64)) {
    return torch::cat(inputs, 0);
  }
  const int64_t columns = first.dim() == 1 ? 1 : first.size(1);
  std::vector<int64_t> offsets(inputs.size() + 1, 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    if (input.device() != first.device() || input.scalar_type() != first.scalar_type() ||
        input.dim() != first.dim() || input.requires_grad() || input.is_neg() || input.is_conj() ||
        (input.dim() == 2 && input.size(1) != columns)) {
      return torch::cat(inputs, 0);
    }
    TORCH_CHECK(input.size(0) <= std::numeric_limits<int64_t>::max() - offsets[i], "row count overflow");
    offsets[i + 1] = offsets[i] + input.size(0);
  }
  const auto options = torch::TensorOptions().dtype(first.scalar_type()).device(torch::kCPU);
  auto output = first.dim() == 1 ? torch::empty({offsets.back()}, options)
                                : torch::empty({offsets.back(), columns}, options);
  const int64_t element_bytes = first.element_size();
  const int64_t row_bytes = columns * element_bytes;
  auto* target = static_cast<char*>(output.data_ptr());
  pybind11::gil_scoped_release release;
  #pragma omp parallel for schedule(static) if(inputs.size() >= 32)
  for (int64_t i = 0; i < static_cast<int64_t>(inputs.size()); ++i) {
    const auto& input = inputs[i];
    if (!input.numel()) continue;
    const auto* source = static_cast<const char*>(input.data_ptr());
    char* destination = target + offsets[i] * row_bytes;
    if (input.is_contiguous()) {
      std::memcpy(destination, source, input.numel() * element_bytes);
      continue;
    }
    const int64_t stride = input.stride(0) * element_bytes;
    const int64_t column_stride = input.dim() == 1 ? element_bytes : input.stride(1) * element_bytes;
    for (int64_t row = 0; row < input.size(0); ++row) {
      if (column_stride == element_bytes) {
        std::memcpy(destination + row * row_bytes, source + row * stride, row_bytes);
      } else {
        for (int64_t column = 0; column < columns; ++column) {
          std::memcpy(destination + row * row_bytes + column * element_bytes,
                      source + row * stride + column * column_stride, element_bytes);
        }
      }
    }
  }
  return output;
}

template <typename scalar_t>
static void expand_ranges(const int64_t* source, const int64_t* lengths,
                          const std::vector<int64_t>& offsets, scalar_t* target) {
  #pragma omp parallel for schedule(static)
  for (int64_t block = 0; block < static_cast<int64_t>(offsets.size()) - 1; ++block) {
    for (int64_t row = 0; row < lengths[block]; ++row) {
      target[offsets[block] + row] = static_cast<scalar_t>(source[block] + row);
    }
  }
}

torch::Tensor expand_cpu_ranges(torch::Tensor starts, torch::Tensor counts, bool int32_output) {
  TORCH_CHECK(starts.device().is_cpu() && counts.device().is_cpu() &&
              starts.scalar_type() == torch::kInt64 && counts.scalar_type() == torch::kInt64 &&
              starts.dim() == 1 && counts.dim() == 1 && starts.numel() == counts.numel() &&
              starts.is_contiguous() && counts.is_contiguous(),
              "starts and counts must be equally sized contiguous CPU int64 vectors");
  const int64_t blocks = starts.numel();
  const int64_t* source = starts.data_ptr<int64_t>();
  const int64_t* lengths = counts.data_ptr<int64_t>();
  std::vector<int64_t> offsets(blocks + 1, 0);
  for (int64_t block = 0; block < blocks; ++block) {
    TORCH_CHECK(source[block] >= 0 && lengths[block] >= 0 &&
                lengths[block] <= std::numeric_limits<int64_t>::max() - source[block] &&
                lengths[block] <= std::numeric_limits<int64_t>::max() - offsets[block],
                "invalid or overflowing range");
    TORCH_CHECK(!int32_output || source[block] + lengths[block] <= std::numeric_limits<int32_t>::max(),
                "range exceeds int32 row IDs");
    offsets[block + 1] = offsets[block] + lengths[block];
  }
  auto output = torch::empty({offsets.back()}, starts.options().dtype(int32_output ? torch::kInt32 : torch::kInt64));
  pybind11::gil_scoped_release release;
  if (int32_output) {
    expand_ranges(source, lengths, offsets, output.data_ptr<int32_t>());
  } else {
    expand_ranges(source, lengths, offsets, output.data_ptr<int64_t>());
  }
  return output;
}

torch::Tensor remap_cpu_rows(torch::Tensor input, torch::Tensor mapping) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == torch::kFloat32 &&
              input.dim() == 2 && input.is_contiguous(),
              "input must be contiguous CPU float32 [N, C]");
  TORCH_CHECK(mapping.device().is_cpu() && mapping.scalar_type() == torch::kInt64 &&
              mapping.dim() == 1 && mapping.is_contiguous(),
              "mapping must be contiguous CPU int64 [M]");
  auto output = torch::empty({mapping.numel(), input.size(1)}, input.options());
  const int64_t columns = input.size(1), input_rows = input.size(0);
  const size_t row_bytes = columns * sizeof(float);
  const float* source = input.data_ptr<float>();
  float* target = output.data_ptr<float>();
  const int64_t* indices = mapping.data_ptr<int64_t>();
  int invalid = 0;
  {
    pybind11::gil_scoped_release release;
    #pragma omp parallel for schedule(static) reduction(|:invalid)
    for (int64_t row = 0; row < mapping.numel(); ++row) {
      const int64_t index = indices[row];
      if (index < -1 || index >= input_rows) {
        invalid = 1;
      } else if (index == -1) {
        std::memset(target + row * columns, 0, row_bytes);
      } else {
        std::memcpy(target + row * columns, source + index * columns, row_bytes);
      }
    }
  }
  TORCH_CHECK(!invalid, "mapping entries must be -1 or a valid input row");
  return output;
}
