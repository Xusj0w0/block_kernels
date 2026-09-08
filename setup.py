from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

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
             str(ROOT / "csrc/cpu_densification.cpp"), str(ROOT / "csrc/cpu_repack.cpp")],
            extra_compile_args={"cxx": ["-O3", "-fopenmp"], "nvcc": ["-O3"]},
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
