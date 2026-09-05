"""CUDA primitives for the BlockGS hybrid transfer backend."""

from __future__ import annotations

import torch

try:
    import _block_kernels_C as _C
except ImportError:  # Optional during CPU-only unit tests.
    _C = None


def _as_cpu_long(value):
    return torch.as_tensor(value, dtype=torch.int64).detach().cpu().flatten()


def dma_copy_h2d_ranges(output, host, source_starts, destination_starts, counts, stream=None):
    """Submit pre-coalesced ranges to CUDA's host-to-device copy engine."""
    source_starts = _as_cpu_long(source_starts)
    destination_starts = _as_cpu_long(destination_starts)
    counts = _as_cpu_long(counts)
    if counts.numel() == 0:
        return
    if stream is None:
        stream = torch.cuda.current_stream(output.device)
    if _C is not None and output.is_cuda and host.is_pinned():
        _C.copy_h2d_ranges(output, host, source_starts, destination_starts, counts, stream.cuda_stream)
        return
    for source, destination, count in zip(source_starts.tolist(), destination_starts.tolist(), counts.tolist()):
        output[destination:destination + count].copy_(host[source:source + count], non_blocking=True)


def dma_copy_d2h_ranges(host, input, source_starts, destination_starts, counts, stream=None):
    """Submit pre-coalesced ranges to CUDA's device-to-host copy engine."""
    source_starts = _as_cpu_long(source_starts)
    destination_starts = _as_cpu_long(destination_starts)
    counts = _as_cpu_long(counts)
    if counts.numel() == 0:
        return
    if stream is None:
        stream = torch.cuda.current_stream(input.device)
    if _C is not None and input.is_cuda and host.is_pinned():
        _C.copy_d2h_ranges(host, input, source_starts, destination_starts, counts, stream.cuda_stream)
        return
    for source, destination, count in zip(source_starts.tolist(), destination_starts.tolist(), counts.tolist()):
        host[destination:destination + count].copy_(input[source:source + count].detach().cpu())


def dma_copy_d2d_ranges(output, input, source_starts, destination_starts, counts, stream=None):
    """Submit pre-coalesced ranges to CUDA's device-to-device copy engine."""
    source_starts = _as_cpu_long(source_starts)
    destination_starts = _as_cpu_long(destination_starts)
    counts = _as_cpu_long(counts)
    if counts.numel() == 0:
        return
    if stream is None:
        stream = torch.cuda.current_stream(output.device)
    if _C is not None and output.is_cuda and input.is_cuda:
        _C.copy_d2d_ranges(output, input, source_starts, destination_starts, counts, stream.cuda_stream)
        return
    for source, destination, count in zip(source_starts.tolist(), destination_starts.tolist(), counts.tolist()):
        output[destination:destination + count].copy_(input[source:source + count])


def mark_compact_mask(
    radii,
    block_ids,
    compact_starts,
    block_offsets,
    block_word_offsets,
    device_mask,
    host_mask,
    finish_word_begin,
    finish_word_end,
    stream=None,
):
    """Mark block-local visibility bits and copy one finish group to CPU."""
    if stream is None:
        stream = torch.cuda.current_stream(radii.device)
    if _C is None:
        raise RuntimeError("the block_kernels CUDA extension is required")
    _C.mark_compact_mask(
        radii.contiguous(),
        block_ids.contiguous(),
        compact_starts.contiguous(),
        block_offsets.contiguous(),
        block_word_offsets.contiguous(),
        device_mask,
        host_mask,
        int(finish_word_begin),
        int(finish_word_end),
        stream.cuda_stream,
    )


def fixed_grid_accumulate_d2h_blocks(host, input, source_starts, destination_starts, counts, stream=None):
    """Accumulate block ranges into mapped pinned host memory with a fixed grid."""
    if _C is None:
        raise RuntimeError("the block_kernels CUDA extension is required")
    if stream is None:
        stream = torch.cuda.current_stream(input.device)
    descriptors = tuple(
        torch.as_tensor(value, dtype=torch.int64, device=input.device).contiguous()
        for value in (source_starts, destination_starts, counts)
    )
    if descriptors[2].numel() == 0:
        return
    _C.fixed_grid_accumulate_d2h_blocks(host, input, *descriptors, stream.cuda_stream)


def extension_available():
    return _C is not None


__all__ = [
    "dma_copy_d2d_ranges",
    "dma_copy_d2h_ranges",
    "dma_copy_h2d_ranges",
    "extension_available",
    "fixed_grid_accumulate_d2h_blocks",
    "mark_compact_mask",
]
