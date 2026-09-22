# Block transfer kernels

The extension copies contiguous table ranges between pinned CPU storage and
GPU buffers. BlockGS uses 11-column geometry DMA before exact visibility, then
loads 48-column SH with `clm_kernels`. Its exact geometry gradients use
`accumulate_d2h_rows`, a mapped-host scatter-add with unique source/destination
row IDs. This avoids assigning an entire CUDA thread block to each 11-value
row. Build it in the `block_gs` environment with:

```bash
python -m pip install --no-build-isolation --no-deps --force-reinstall submodules/block_kernels
```

The compiled extension is required for training. The Python DMA fallback is
intended for diagnostics; mapped-host gradient accumulation requires CUDA.

The same extension now owns the C++/OpenMP LoD hierarchy initializer. Its
`initialize_lod_arrays` and `initialize_lod_ply` bindings are wrapped by
`block_kernels.initialize_blocks`; LoD initialization no longer performs a
separate runtime JIT build. The PLY path uses per-block byte offsets and
`pread`; the array path accepts fixed-slot block starts and may safely rewrite
the same leaf tables in place because each worker copies its complete source
block to private scratch before producing output.

The extension exposes DMA `copy_h2d_ranges`, `copy_d2h_ranges`, and
`copy_d2d_ranges`, plus `fixed_grid_accumulate_d2h_blocks` for fixed-grid mapped-host
gradient accumulation.

`mark_compact_mask` converts block-contiguous projection radii into a compact,
word-aligned union bitset on GPU and copies only the current finish group's
contiguous word range to pinned CPU memory.

`update_metadata_from_geometry` reduces full staged 11-column blocks at their
last coarse-visible camera. CUDA descriptors contain `(block ID, staged start,
count)`; the caller guarantees unique block IDs and valid ranges. The kernel
updates AABBs, centers, and radii in place on the supplied prefetch stream,
including three-sigma scale extents. It allocates no point-sized temporary and
does not transfer additional geometry. `cpu_block_bounds` remains available for
initialization and explicit full layout construction.

`remap_cpu_rows` gathers a float32 CPU state table by a CPU int64 source map,
with `-1` producing a zero row. It avoids a full zero pass, a gathered temporary
state table, and masked scatter. `expand_cpu_ranges` writes CPU int64 point IDs
directly from `(start, count)` ranges without point-sized intermediate maps.
Both are CPU-only OpenMP kernels; neither allocates CUDA tensors.

`repack_cpu_blocks(tables, order, counts, block_size)` applies a unique int64
destination-to-source order to geometry, SH and optional Adam moment tables in
place. It validates that the mapping covers exactly the live block prefixes,
then handles permutation paths and cycles using bitsets and one scratch row per
table. Tables run in parallel and must have disjoint contiguous float32 CPU
storage with the same capacity. It clears vacated assigned slots and returns
full-block counts (except the last block) and fresh three-sigma AABB bounds.
Preallocated table addresses remain unchanged.

`concat_cpu_rows` copies large lists of CPU float32/int64 block views in parallel
to ordinary unpinned CPU storage. It supports strided rows/columns and delegates
unsupported dtypes, type promotion, and autograd inputs to `torch.cat`.
