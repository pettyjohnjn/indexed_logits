"""
build_inline.py

Alternative build method using torch.utils.cpp_extension.load_inline.
This compiles the extension at runtime without needing setup.py.

Usage:
    python build_inline.py

    # This will compile and run a quick test.
    # The compiled extension is cached in ~/.cache/torch_extensions/

After running, you can import the module:
    from build_inline import indexed_logits, IndexedLogits
"""


import re
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_HERE = Path(__file__).resolve().parent

# Kernel sources: read the canonical .cpp/.cu (the single source of truth shared
# with the setup.py build) instead of embedding copies here. The standalone .cpp
# binds via PYBIND11_MODULE for the setup.py build; load_inline generates its own
# bindings from `functions=` below, so we strip that block when reading.
cpp_source = re.sub(
    r"\nPYBIND11_MODULE.*", "",
    (_HERE / "indexed_logits.cpp").read_text(),
    flags=re.DOTALL,
)
cuda_source = (_HERE / "indexed_logits_cuda.cu").read_text()

# Build and load the extension

print("Compiling indexed_logits CUDA extension...")
print("This may take a minute on first run (cached afterwards).")

indexed_logits_cuda = load_inline(
    name='indexed_logits_cuda',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['indexed_logits_forward', 'indexed_logits_backward'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    extra_cflags=['-O3'],
)

print("Compilation successful!")

# Python wrapper (same as indexed_logits.py)

from torch.autograd import Function


def indexed_logits_forward(H, W, idx):
    """Forward: out[i,j] = dot(H[i,:], W[idx[i,j],:])"""
    return indexed_logits_cuda.indexed_logits_forward(H, W, idx)

def indexed_logits_backward(H, W, idx, grad_out):
    """Backward: compute grad_H and grad_W"""
    return indexed_logits_cuda.indexed_logits_backward(H, W, idx, grad_out)

class IndexedLogits(Function):
    """Autograd Function for indexed logits."""

    @staticmethod
    def forward(ctx, H, W, idx):
        if idx.dtype == torch.int64:
            idx = idx.to(torch.int32)
        ctx.save_for_backward(H, W, idx)
        return indexed_logits_cuda.indexed_logits_forward(H, W, idx)

    @staticmethod
    def backward(ctx, grad_out):
        H, W, idx = ctx.saved_tensors
        grad_H, grad_W = indexed_logits_cuda.indexed_logits_backward(H, W, idx, grad_out)
        return grad_H, grad_W, None

def indexed_logits(H, W, idx):
    """
    Compute indexed logits: out[i,j] = dot(H[i,:], W[idx[i,j],:])

    Memory-efficient: does NOT materialize W[idx] as [N, k, d].
    """
    return IndexedLogits.apply(H, W, idx)

# Quick test

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running quick validation test...")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    # Small test
    N, d, V, k = 32, 64, 100, 8

    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=torch.float16, device='cuda', requires_grad=True)
    W = torch.randn(V, d, dtype=torch.float16, device='cuda', requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')

    # Reference
    H_ref = H.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    out_ref = (H_ref.unsqueeze(1) * W_ref[idx]).sum(-1)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # Our implementation
    out = indexed_logits(H, W, idx)
    loss = out.sum()
    loss.backward()

    # Check
    print(f"\nForward max error: {(out - out_ref).abs().max().item():.6e}")
    print(f"grad_H max error: {(H.grad - H_ref.grad).abs().max().item():.6e}")
    print(f"grad_W max error: {(W.grad - W_ref.grad).abs().max().item():.6e}")

    fwd_ok = torch.allclose(out, out_ref, rtol=1e-2, atol=1e-2)
    grad_H_ok = torch.allclose(H.grad, H_ref.grad, rtol=1e-2, atol=1e-2)
    grad_W_ok = torch.allclose(W.grad, W_ref.grad, rtol=1e-2, atol=1e-2)

    print(f"\nForward: {'PASS' if fwd_ok else 'FAIL'}")
    print(f"grad_H: {'PASS' if grad_H_ok else 'FAIL'}")
    print(f"grad_W: {'PASS' if grad_W_ok else 'FAIL'}")

    if fwd_ok and grad_H_ok and grad_W_ok:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
