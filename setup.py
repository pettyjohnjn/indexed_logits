"""
setup.py

Build script for the indexed_logits CUDA extension.

Usage:
    # Install globally:
    pip install .
    
    # Or build in-place for development:
    python setup.py build_ext --inplace
    
    # Or install in development mode:
    pip install -e .

Requirements:
    - PyTorch with CUDA support
    - CUDA toolkit (matching PyTorch's CUDA version)
    - C++ compiler (gcc on Linux, MSVC on Windows)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get CUDA architecture flags
# This ensures compatibility with the current GPU
def get_cuda_arch_flags():
    """Get CUDA architecture flags for compilation."""
    # Let PyTorch figure out the architectures
    # This will use TORCH_CUDA_ARCH_LIST if set, or detect from the current GPU
    return []

# Extra compile arguments for optimization
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--use_fast_math',  # Fast math operations
        '-lineinfo',  # Line info for profiling
    ]
}

# Add architecture flags if available
cuda_arch_list = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
for arch in cuda_arch_list:
    # Extract compute capability number (e.g., "sm_80" -> "80")
    if arch.startswith('sm_'):
        cc = arch[3:]
        extra_compile_args['nvcc'].extend(['-gencode', f'arch=compute_{cc},code=sm_{cc}'])

setup(
    name='indexed_logits_cuda',
    version='1.0.0',
    description='Memory-efficient indexed logits computation for transformer models',
    author='Your Name',
    ext_modules=[
        CUDAExtension(
            name='indexed_logits_cuda',
            sources=[
                'indexed_logits.cpp',
                'indexed_logits_cuda.cu',
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
    ],
    py_modules=['indexed_logits'],
)
