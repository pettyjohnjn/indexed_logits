"""
setup.py

Build script for the indexed_logits CUDA extension.

Usage:
    pip install .
    pip install -e .
    python setup.py build_ext --inplace
"""

import sys
from setuptools import setup


def is_metadata_cmd(argv):
    # Commands that should not import torch (pip may run these on login nodes / build frontends)
    return any(cmd in argv for cmd in ("egg_info", "dist_info", "--name", "--version"))


# IMPORTANT: this project provides a single Python module file: indexed_logits.py
# It is NOT a package directory named "indexed_logits/".
# Therefore we must use py_modules, not packages.
if is_metadata_cmd(sys.argv):
    setup(
        name="indexed_logits_cuda",
        version="1.0.0",
        py_modules=["indexed_logits"],
        install_requires=["torch>=2.0.0"],
    )
    raise SystemExit(0)

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="indexed_logits_cuda",
    version="1.0.0",
    py_modules=["indexed_logits"],
    ext_modules=[
        CUDAExtension(
            name="indexed_logits_cuda",
            sources=["indexed_logits.cpp", "indexed_logits_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.0.0"],
)