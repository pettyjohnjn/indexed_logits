# Indexed Logits CUDA Extension

A memory-efficient PyTorch CUDA extension for computing indexed logits without materializing large intermediate tensors.

## Overview

This extension computes:
```
out[i,j] = dot(H[i,:], W[idx[i,j],:])
```

for:
- `i` in `[0, N-1]`
- `j` in `[0, k-1]`

**Key Feature**: Unlike the naive implementation `(H.unsqueeze(1) * W[idx]).sum(-1)`, this extension does NOT allocate an `[N, k, d]` intermediate tensor, saving significant GPU memory for large models.

## Installation

### Option 1: pip install (recommended)

```bash
cd indexed_logits
pip install .
```

### Option 2: Build in-place

```bash
cd indexed_logits
python setup.py build_ext --inplace
```

### Option 3: Load inline (no installation needed)

```python
# Just run build_inline.py - it compiles and caches automatically
from build_inline import indexed_logits
```

## Usage

```python
import torch
from indexed_logits import indexed_logits

# Setup
N, d, V, k = 4096, 1024, 50000, 64
H = torch.randn(N, d, dtype=torch.float16, device='cuda', requires_grad=True)
W = torch.randn(V, d, dtype=torch.float16, device='cuda', requires_grad=True)
idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')

# Forward
out = indexed_logits(H, W, idx)  # [N, k]

# Backward (automatic)
loss = out.sum()
loss.backward()

print(H.grad.shape)  # [N, d]
print(W.grad.shape)  # [V, d]
```

## API Reference

### `indexed_logits(H, W, idx) -> Tensor`

Main entry point with automatic differentiation support.

**Arguments:**
- `H`: Hidden states tensor, shape `[N, d]`, dtype `float16` or `bfloat16`, CUDA
- `W`: Weight matrix, shape `[V, d]`, same dtype as `H`, CUDA
- `idx`: Token indices, shape `[N, k]`, dtype `int32` (int64 accepted but converted), CUDA

**Returns:**
- `out`: Logits tensor, shape `[N, k]`, same dtype as `H`

### `indexed_logits_forward(H, W, idx) -> Tensor`

Direct forward pass without autograd.

### `indexed_logits_backward(H, W, idx, grad_out) -> (grad_H, grad_W)`

Direct backward pass. Note: `grad_W` is returned in `float32` for numerical stability.

### `IndexedLogits` (torch.autograd.Function)

The underlying autograd Function class for advanced usage.

## Implementation Details

### Forward Pass
Each CUDA thread computes one output element by iterating over the `d` dimension:
```
out[i,j] = sum_{t=0}^{d-1} H[i,t] * W[idx[i,j], t]
```
Dot products are accumulated in float32 for numerical stability, then cast to output dtype.

### Backward Pass

**grad_H**: Each thread computes one element `grad_H[i,t]`:
```
grad_H[i,t] = sum_{j=0}^{k-1} grad_out[i,j] * W[idx[i,j], t]
```

**grad_W**: Uses atomic adds in float32 to handle collisions:
```
grad_W[token,t] += sum_{(i,j): idx[i,j]=token} grad_out[i,j] * H[i,t]
```

Note: `grad_W` is returned in `float32` because:
1. Atomic adds in fp16/bf16 have limited hardware support
2. Float32 accumulation provides better numerical stability
3. PyTorch autograd accepts mixed-precision gradients

## Testing

```bash
python test_indexed_logits.py
```

The test suite validates:
- Forward correctness against reference implementation
- Backward gradients for both `grad_H` and `grad_W`
- Support for `float16` and `bfloat16`
- Edge cases (k=1, large dimensions, repeated indices)
- Memory efficiency verification

## Files

```
indexed_logits/
├── indexed_logits.cpp      # PyBind11 bindings and validation
├── indexed_logits_cuda.cu  # CUDA kernels
├── indexed_logits.py       # Python wrapper with autograd.Function
├── setup.py                # Build script (CUDAExtension)
├── build_inline.py         # Alternative: load_inline compilation
├── test_indexed_logits.py  # Test suite
└── README.md               # This file
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0 with CUDA support
- CUDA toolkit (matching PyTorch's CUDA version)
- C++ compiler (gcc on Linux)

## Performance Notes

This implementation prioritizes **correctness over speed**. Potential optimizations (not implemented):
- Vectorized loads (float4)
- Shared memory for H rows
- Warp-level reductions
- Tiling for better cache utilization

The atomic operations in `grad_W` may cause contention when many indices map to the same token. This is inherent to the problem and acceptable for correctness-first design.
