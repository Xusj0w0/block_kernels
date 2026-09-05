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
            [str(ROOT / "csrc/block_transfer.cu"), str(ROOT / "csrc/ext.cpp")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
