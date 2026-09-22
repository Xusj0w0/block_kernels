"""Python validation and dispatch for block-local LoD initialization."""

import numpy as np
import torch
from utils.general_utils import GaussianPly

LOD_INITIALIZER = "cpp_openmp_hgs_area_v1"


def backend():
    from . import _C
    if _C is None or not hasattr(_C, "initialize_lod_arrays"):
        raise RuntimeError("rebuild block_kernels with LoD initialization support")
    return _C


def initialize_blocks(geometry, sh, offsets, block_size, levels=(4, 6), *,
                      outputs=None, leaf_base=0, source_starts=None, order_starts=None,
                      threads=None, progress_every=0):
    """Write packed hierarchy tables directly into caller-owned CPU buffers.

    A native float32 binary GaussianPly source is read directly by OpenMP
    workers into fixed single-block buffers for the whole call (pass sh=None).
    progress_every controls completed-block logging, with 0 disabling it. Contiguous
    NumPy batches use the same arithmetic, also supporting non-native PLY.
    """
    levels = tuple(levels)
    if len(levels) != 2 or not 1 <= levels[0] < levels[1] or levels[1] > 30:
        raise ValueError("require two increasing positive merge depths")
    S, span, coarse_span = int(block_size), 1 << levels[0], 1 << levels[1]
    if S < coarse_span or S & (S-1) or S % coarse_span:
        raise ValueError("block size must be a power of two divisible by merge spans")
    if (not isinstance(offsets, np.ndarray) or offsets.dtype != np.int64
            or not offsets.flags.c_contiguous):
        raise ValueError("offsets must be contiguous int64")
    if offsets.ndim != 1 or len(offsets) < 1:
        raise ValueError("offsets must be a nonempty vector")
    if (offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1])
            or np.any(offsets[1:] - offsets[:-1] > S)):
        raise ValueError('invalid block counts/prefix offsets')
    source = geometry if isinstance(geometry, GaussianPly) else None
    disk_source = source is not None
    if source is not None:
        if sh is not None or source.degree != 3 or leaf_base < 0 or leaf_base+offsets[-1] > source.rows:
            raise ValueError('invalid PLY source or row range')
        native = (source.array is None and source.dtype.itemsize == 4*len(source.dtype.names)
                  and all(dtype == np.dtype('float32') for dtype,_ in source.dtype.fields.values()))
        if not native:
            geometry, sh = source.read(leaf_base, int(offsets[-1]))
            source = None
    if source is None:
        if (not isinstance(geometry, np.ndarray) or not isinstance(sh, np.ndarray)
                or geometry.dtype != np.float32 or sh.dtype != np.float32
                or geometry.ndim != 2 or geometry.shape[1] != 11 or sh.shape != (len(geometry),48)
                or not geometry.flags.c_contiguous or not sh.flags.c_contiguous):
            raise ValueError('expected contiguous float32 geometry [N,11] and SH [N,48]')
    B, P = len(offsets)-1, S//span+S//coarse_span
    if source_starts is None:
        read_starts = offsets[:-1].copy()
        if disk_source and source is not None:
            read_starts += leaf_base
    else:
        read_starts = np.asarray(source_starts)
        if (read_starts.dtype != np.int64 or read_starts.shape != (B,)
                or not read_starts.flags.c_contiguous):
            raise ValueError('source_starts must be contiguous int64 with one entry per block')
        read_starts = read_starts.copy()
    if order_starts is None:
        order_starts = offsets[:-1].copy()
        order_starts += leaf_base
    else:
        order_starts = np.asarray(order_starts)
        if (order_starts.dtype != np.int64 or order_starts.shape != (B,)
                or not order_starts.flags.c_contiguous):
            raise ValueError('order_starts must be contiguous int64 with one entry per block')
        order_starts = order_starts.copy()
    shapes = {"leaf_geometry": (B*S, 11), "leaf_sh": (B*S, 48),
              "proxy_geometry": (B*P, 11), "proxy_sh": (B*P, 48),
              "leaf_order": (B, S), "node_counts": (B, S+P),
              "minimum": (B, 3), "maximum": (B, 3)}
    dtypes = {"leaf_order": np.int64, "node_counts": np.int32}
    if outputs is None:
        outputs = {name: np.empty(shape, dtype=dtypes.get(name, np.float32))
                   for name, shape in shapes.items()}
    for name, shape in shapes.items():
        value = outputs[name]
        if (value.shape != shape or value.dtype != dtypes.get(name, np.float32)
                or not value.flags.c_contiguous or not value.flags.writeable):
            raise ValueError(f"invalid initialization buffer: {name}")
    if source is None:
        for value, output_name in ((geometry, 'leaf_geometry'), (sh, 'leaf_sh')):
            if np.shares_memory(value, outputs[output_name]):
                expected = np.arange(B, dtype=np.int64) * S
                same_start = (value.__array_interface__['data'][0]
                              == outputs[output_name].__array_interface__['data'][0])
                if not same_start or not np.array_equal(read_starts, expected):
                    raise ValueError('in-place initialization requires fixed block-aligned source tables')
    common = (offsets, read_starts, order_starts, S, span, coarse_span,
              torch.get_num_threads() if threads is None else threads, progress_every,
              *(outputs[name] for name in shapes))
    if source is not None:
        columns = list(source.dtype.names)
        geometry_names = ['x','y','z','opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
        sh_names = [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{channel*15+coeff}' for coeff in range(15) for channel in range(3)]
        gcols = np.array([columns.index(n) for n in geometry_names], dtype=np.int64)
        hcols = np.array([columns.index(n) for n in sh_names], dtype=np.int64)
        # A single shared address table, prepared before any worker starts.
        # Workers only look up their block's byte address and valid row count.
        byte_offsets = read_starts.copy()
        byte_offsets *= source.dtype.itemsize
        byte_offsets += source.offset
        backend().initialize_lod_ply(source.file.fileno(), source.offset, len(columns),
                                     gcols, hcols, source.rows, byte_offsets, *common)
    else:
        backend().initialize_lod_arrays(geometry, sh, *common)
    return outputs
