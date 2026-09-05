# Block transfer kernels

The extension copies complete packed BlockGS `[rows, 59]` ranges directly
between pinned CPU storage and GPU staging buffers. Adjacent ranges are merged
before launch. Gradient fragments at a block's first eviction use D2H overwrite;
only non-consecutive reappearances add directly into mapped pinned host rows,
while retained gradients use device-to-device range copies. Build
it in the `block_gs` environment with:

```bash
conda run -n block_gs python -m pip install -e ./block_kernels --no-build-isolation
```

`block_kernels/__init__.py` provides an equivalent PyTorch fallback for unit
tests and systems where the extension has not been compiled. The fallback is
functionally correct but synchronizes GPU-to-host gradient accumulation; build
the extension to overlap retained-gradient transfers and CPU Adam with
rendering.

The extension exposes DMA `copy_h2d_ranges`, `copy_d2h_ranges`, and
`copy_d2d_ranges`, plus `fixed_grid_accumulate_d2h_blocks` for fixed-grid mapped-host
gradient accumulation.

`mark_compact_mask` converts block-contiguous projection radii into a compact,
word-aligned union bitset on GPU and copies only the current finish group's
contiguous word range to pinned CPU memory.
