from pathlib import Path
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
EIGEN_INCLUDE = next(
    (
        path
        for path in (
            os.environ.get("EIGEN3_INCLUDE_DIR", ""),
            str(Path(sys.prefix) / "include/eigen3"),
            "/usr/include/eigen3",
        )
        if path and (Path(path) / "Eigen/Dense").is_file()
    ),
    None,
)
if EIGEN_INCLUDE is None:
    raise RuntimeError(
        "Eigen3 headers required: install libeigen3-dev or set EIGEN3_INCLUDE_DIR"
    )

setup(
    name="block_kernels",
    packages=["block_kernels"],
    package_dir={"block_kernels": "."},
    ext_modules=[
        CUDAExtension(
            "_block_kernels_C",
            [str(ROOT / "csrc/block_transfer.cu"), str(ROOT / "csrc/ext.cpp"),
             str(ROOT / "csrc/block_metadata.cpp"), str(ROOT / "csrc/gpu_block_metadata.cu"),
             str(ROOT / "csrc/cpu_row_mapping.cpp"), str(ROOT / "csrc/cpu_block_tasks.cpp"),
             str(ROOT / "csrc/cpu_densification.cpp"), str(ROOT / "csrc/cpu_repack.cpp"),
             str(ROOT / "csrc/lod_initialize.cpp")],
            include_dirs=[EIGEN_INCLUDE],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp", "-DEIGEN_DONT_PARALLELIZE", "-ffp-contract=off"],
                "nvcc": ["-O3"],
            },
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
