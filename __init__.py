"""CUDA primitives for the BlockGS hybrid transfer backend."""

from __future__ import annotations

import torch
from typing import NamedTuple

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


def accumulate_d2h_rows(host, input, source, destination, stream=None):
    """Scatter-add unique exact-visible rows to a contiguous pinned table."""
    if _C is None or not hasattr(_C, "accumulate_d2h_rows"):
        raise RuntimeError("rebuild block_kernels for the exact-row transfer kernel")
    stream = stream or torch.cuda.current_stream(input.device)
    _C.accumulate_d2h_rows(host, input, source, destination, stream.cuda_stream)


def extension_available():
    return _C is not None


def cpu_block_bounds(points, radii, offsets):
    """Parallel CPU bounds for contiguous blocks, including strided geometry views."""
    if _C is None or not hasattr(_C, "cpu_block_bounds"):
        return None
    if radii is None:
        radii = torch.empty(0, dtype=points.dtype)
    return _C.cpu_block_bounds(points, radii, offsets)


def remap_cpu_rows(input, mapping):
    """Gather CPU state rows in one pass, zeroing only newly inserted rows (-1)."""
    if _C is None or not hasattr(_C, "remap_cpu_rows"):
        raise RuntimeError("rebuild block_kernels for CPU state remapping")
    return _C.remap_cpu_rows(input, mapping)


def expand_cpu_ranges(starts, counts, dtype=torch.int64):
    """Expand block ranges without point-sized intermediate indexing tensors."""
    if _C is None or not hasattr(_C, "expand_cpu_ranges"):
        return None
    if dtype not in (torch.int64, torch.int32):
        raise ValueError("range output must be int32 or int64")
    return _C.expand_cpu_ranges(starts, counts, dtype == torch.int32)


def concat_cpu_rows(inputs):
    """Concatenate CPU block views into unpinned storage with parallel copies."""
    if _C is None or not hasattr(_C, "concat_cpu_rows"):
        return torch.cat(inputs, dim=0)
    return _C.concat_cpu_rows(inputs)


def commit_cpu_block_tasks(tables, mappings, geometry, sh, destinations, block_size):
    """Apply disjoint fixed-slot edits in parallel, preserving moved Adam moments."""
    if _C is None or not hasattr(_C, "commit_cpu_block_tasks"):
        raise RuntimeError("rebuild block_kernels for parallel CPU block tasks")
    _C.commit_cpu_block_tasks(tables, mappings, geometry, sh, destinations, block_size)


def repack_cpu_blocks(tables, order, counts, block_size):
    """Reorder parameters and moments in place, pack full blocks and rebuild bounds."""
    if _C is None or not hasattr(_C, "repack_cpu_blocks"):
        raise RuntimeError("rebuild block_kernels for final Morton repacking")
    return _C.repack_cpu_blocks(tables, order, counts, block_size)


class DensityBlockPlan(NamedTuple):
    counts: torch.Tensor
    overflow_ids: torch.Tensor
    selected_bits: torch.Tensor
    source_blocks: torch.Tensor
    source_offsets: torch.Tensor
    destination_blocks: torch.Tensor
    destination_offsets: torch.Tensor
    final_counts: torch.Tensor
    expansion_ids: torch.Tensor
    expansion_min: torch.Tensor
    expansion_max: torch.Tensor
    timings: torch.Tensor
    metrics: torch.Tensor


def densify_cpu_stage1(tables, gradients, counts, block_size, threshold, dense_scale):
    """Edit nonoverflow blocks and return overflow IDs, bitsets and updated counts."""
    if _C is None or not hasattr(_C, "densify_cpu_stage1"):
        raise RuntimeError("rebuild block_kernels for C++ densification")
    return DensityBlockPlan(*_C.densify_cpu_stage1(tables, gradients, counts, block_size, threshold, dense_scale))


def densify_cpu_stage2(tables, plan, block_size, dense_scale, min_opacity, world_scale):
    """Consume stage-one overflow selections, repartition clusters and prune on CPU."""
    if _C is None or not hasattr(_C, "densify_cpu_stage2"):
        raise RuntimeError("rebuild block_kernels for C++ densification")
    return _C.densify_cpu_stage2(tables, plan, block_size, dense_scale, min_opacity, world_scale)


def update_metadata_from_geometry(geometry, ranges, aabb_min, aabb_max, centers, radii, stream=None):
    """Reduce full staged blocks; ranges are unique (block ID, staged start, count).

    Ranges must address valid geometry and metadata rows. The caller constructs
    them from the CPU layout to avoid a device synchronization for validation.
    """
    if _C is None or not hasattr(_C, "update_metadata_from_geometry"):
        raise RuntimeError("rebuild block_kernels for GPU metadata updates")
    stream = stream or torch.cuda.current_stream(geometry.device)
    _C.update_metadata_from_geometry(geometry, ranges, aabb_min, aabb_max, centers, radii, stream.cuda_stream)


__all__ = [
    "dma_copy_d2d_ranges",
    "dma_copy_d2h_ranges",
    "dma_copy_h2d_ranges",
    "extension_available",
    "fixed_grid_accumulate_d2h_blocks",
    "accumulate_d2h_rows",
    "cpu_block_bounds",
    "remap_cpu_rows",
    "expand_cpu_ranges",
    "concat_cpu_rows",
    "commit_cpu_block_tasks",
    "repack_cpu_blocks",
    "densify_cpu_stage1",
    "densify_cpu_stage2",
    "update_metadata_from_geometry",
    "mark_compact_mask",
]
