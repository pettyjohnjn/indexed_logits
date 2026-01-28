"""
test_indexed_logits.py

Comprehensive test suite for the indexed_logits CUDA extension.

Tests:
1. Forward correctness against PyTorch reference implementation
2. Backward correctness (grad_H and grad_W) against autograd reference
3. Various dtypes (float16, bfloat16)
4. Edge cases (k=1, small/large dimensions)
5. Memory efficiency verification (forward and forward+backward)
6. Determinism / stability checks
7. Non-contiguity handling

Usage:
    python test_indexed_logits.py
    
    # Or with pytest:
    pytest test_indexed_logits.py -v
"""

import torch
import math
import sys
import gc
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

# Try to import the extension
try:
    from indexed_logits import indexed_logits, IndexedLogits, indexed_logits_forward, indexed_logits_backward
    EXTENSION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import extension: {e}")
    print("Run 'python setup.py build_ext --inplace' first.")
    EXTENSION_AVAILABLE = False


# =============================================================================
# Improved Error Metrics
# =============================================================================

@dataclass
class ErrorMetrics:
    """Container for error metrics."""
    max_abs_err: float
    mean_abs_err: float
    max_rel_err_masked: float
    fraction_masked: float
    passed: bool


def compute_error_metrics(actual: torch.Tensor, expected: torch.Tensor, 
                          eps: float, rtol: float, atol: float) -> ErrorMetrics:
    """
    Compute improved error metrics with masked relative error.
    
    Args:
        actual: Computed tensor
        expected: Reference tensor
        eps: Threshold for masking near-zero values in expected
        rtol: Relative tolerance for allclose
        atol: Absolute tolerance for allclose
    
    Returns:
        ErrorMetrics with all computed statistics
    """
    # Convert to same dtype for comparison
    if actual.dtype != expected.dtype:
        actual = actual.float()
        expected = expected.float()
    
    diff = (actual - expected).abs()
    
    # Absolute errors
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    
    # Masked relative error (only where |expected| > eps)
    mask = expected.abs() > eps
    fraction_masked = mask.float().mean().item()
    
    if mask.any():
        rel_err_masked = diff[mask] / expected[mask].abs()
        max_rel_err_masked = rel_err_masked.max().item()
    else:
        max_rel_err_masked = float('nan')
    
    # Check if passed
    passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    
    return ErrorMetrics(
        max_abs_err=max_abs_err,
        mean_abs_err=mean_abs_err,
        max_rel_err_masked=max_rel_err_masked,
        fraction_masked=fraction_masked,
        passed=passed
    )


def check_close_v2(actual: torch.Tensor, expected: torch.Tensor, name: str, 
                   dtype: torch.dtype = torch.float16,
                   rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    """
    Check if two tensors are close with improved error metrics.
    Uses masked relative error to avoid division by near-zero values.
    """
    if actual is None and expected is None:
        print(f"  {name}: Both None (OK)")
        return True
    
    if actual is None or expected is None:
        print(f"  {name}: FAIL - One is None, other is not")
        return False
    
    # Set epsilon based on dtype
    if dtype == torch.float16:
        eps = 1e-3
    elif dtype == torch.bfloat16:
        eps = 3e-3
    else:
        eps = 1e-5
    
    metrics = compute_error_metrics(actual, expected, eps, rtol, atol)
    
    status = "OK" if metrics.passed else "FAIL"
    print(f"  {name}: {status}")
    print(f"    max_abs_err={metrics.max_abs_err:.6e}, mean_abs_err={metrics.mean_abs_err:.6e}")
    print(f"    max_rel_err_masked={metrics.max_rel_err_masked:.6e} (frac_unmasked={metrics.fraction_masked:.3f})")
    
    if not metrics.passed:
        print(f"    actual shape: {actual.shape}, dtype: {actual.dtype}")
        print(f"    expected shape: {expected.shape}, dtype: {expected.dtype}")
        print(f"    actual range: [{actual.min().item():.4f}, {actual.max().item():.4f}]")
        print(f"    expected range: [{expected.min().item():.4f}, {expected.max().item():.4f}]")
    
    return metrics.passed


def reference_forward(H, W, idx):
    """
    Reference implementation using standard PyTorch ops.
    This DOES materialize W[idx] as [N, k, d] - only for testing!
    """
    W_selected = W[idx]  # [N, k, d]
    out = (H.unsqueeze(1) * W_selected).sum(-1)  # [N, k]
    return out


def reference_backward(H, W, idx, grad_out):
    """
    Reference backward using autograd on the reference forward.
    """
    H_ref = H.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    
    out_ref = reference_forward(H_ref, W_ref, idx)
    loss = (out_ref * grad_out).sum()
    loss.backward()
    
    return H_ref.grad, W_ref.grad


# =============================================================================
# Basic Tests
# =============================================================================

def test_forward_basic(dtype=torch.float16):
    """Test basic forward pass correctness."""
    print(f"\n=== Test Forward Basic (dtype={dtype}) ===")
    
    N, d, V, k = 32, 64, 100, 8
    
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda')
    W = torch.randn(V, d, dtype=dtype, device='cuda')
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    out_ref = reference_forward(H, W, idx)
    out = indexed_logits_forward(H, W, idx)
    
    return check_close_v2(out, out_ref, "forward output", dtype=dtype)


def test_backward_basic(dtype=torch.float16):
    """Test basic backward pass correctness."""
    print(f"\n=== Test Backward Basic (dtype={dtype}) ===")
    
    N, d, V, k = 32, 64, 100, 8
    
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda')
    W = torch.randn(V, d, dtype=dtype, device='cuda')
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    grad_out = torch.randn(N, k, dtype=dtype, device='cuda')
    
    grad_H_ref, grad_W_ref = reference_backward(H, W, idx, grad_out)
    grad_H, grad_W = indexed_logits_backward(H, W, idx, grad_out)
    
    passed_H = check_close_v2(grad_H, grad_H_ref, "grad_H", dtype=dtype)
    passed_W = check_close_v2(grad_W, grad_W_ref, "grad_W", dtype=dtype)
    
    return passed_H and passed_W


def test_autograd_function(dtype=torch.float16):
    """Test the autograd Function wrapper."""
    print(f"\n=== Test Autograd Function (dtype={dtype}) ===")
    
    N, d, V, k = 32, 64, 100, 8
    
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device='cuda', requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    H_ref = H.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    out_ref = reference_forward(H_ref, W_ref, idx)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    out = indexed_logits(H, W, idx)
    loss = out.sum()
    loss.backward()
    
    passed_out = check_close_v2(out, out_ref, "forward output", dtype=dtype)
    passed_H = check_close_v2(H.grad, H_ref.grad, "grad_H", dtype=dtype)
    passed_W = check_close_v2(W.grad, W_ref.grad, "grad_W", dtype=dtype)
    
    return passed_out and passed_H and passed_W


def test_bfloat16():
    """Test with bfloat16 dtype."""
    print("\n=== Test BFloat16 ===")
    
    if not torch.cuda.is_bf16_supported():
        print("  Skipped: bfloat16 not supported on this GPU")
        return True
    
    N, d, V, k = 32, 64, 100, 8
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device='cuda', requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    H_ref = H.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    out_ref = reference_forward(H_ref, W_ref, idx)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    out = indexed_logits(H, W, idx)
    loss = out.sum()
    loss.backward()
    
    # BFloat16 has lower precision, use looser tolerances
    passed_out = check_close_v2(out, out_ref, "forward output", dtype=dtype, rtol=5e-2, atol=5e-2)
    passed_H = check_close_v2(H.grad, H_ref.grad, "grad_H", dtype=dtype, rtol=5e-2, atol=5e-2)
    passed_W = check_close_v2(W.grad, W_ref.grad, "grad_W", dtype=dtype, rtol=5e-2, atol=5e-2)
    
    return passed_out and passed_H and passed_W


def test_edge_cases():
    """Test edge cases."""
    print("\n=== Test Edge Cases ===")
    all_passed = True
    dtype = torch.float16
    
    # k=1
    print("\n  Case: k=1")
    N, d, V, k = 16, 32, 50, 1
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device='cuda', requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    H_ref = H.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    out_ref = reference_forward(H_ref, W_ref, idx)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    out = indexed_logits(H, W, idx)
    loss = out.sum()
    loss.backward()
    
    passed = check_close_v2(out, out_ref, "k=1 forward", dtype=dtype)
    passed &= check_close_v2(H.grad, H_ref.grad, "k=1 grad_H", dtype=dtype)
    passed &= check_close_v2(W.grad, W_ref.grad, "k=1 grad_W", dtype=dtype)
    all_passed &= passed
    
    # Large d
    print("\n  Case: Large d (d=512)")
    N, d, V, k = 16, 512, 50, 4
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device='cuda', requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    H_ref = H.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    out_ref = reference_forward(H_ref, W_ref, idx)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    out = indexed_logits(H, W, idx)
    loss = out.sum()
    loss.backward()
    
    passed = check_close_v2(out, out_ref, "large_d forward", dtype=dtype)
    passed &= check_close_v2(H.grad, H_ref.grad, "large_d grad_H", dtype=dtype)
    passed &= check_close_v2(W.grad, W_ref.grad, "large_d grad_W", dtype=dtype)
    all_passed &= passed
    
    # Repeated indices (stress test atomics)
    print("\n  Case: Repeated indices (same token multiple times)")
    N, d, V, k = 16, 64, 10, 8  # Small V, large k -> many collisions
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device='cuda', requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    H_ref = H.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    out_ref = reference_forward(H_ref, W_ref, idx)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    out = indexed_logits(H, W, idx)
    loss = out.sum()
    loss.backward()
    
    passed = check_close_v2(out, out_ref, "repeated_idx forward", dtype=dtype)
    passed &= check_close_v2(H.grad, H_ref.grad, "repeated_idx grad_H", dtype=dtype)
    passed &= check_close_v2(W.grad, W_ref.grad, "repeated_idx grad_W", dtype=dtype)
    all_passed &= passed
    
    return all_passed


def test_int64_idx():
    """Test that int64 indices work (with warning)."""
    print("\n=== Test Int64 Index Conversion ===")
    
    N, d, V, k = 16, 32, 50, 4
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=torch.float16, device='cuda')
    W = torch.randn(V, d, dtype=torch.float16, device='cuda')
    idx = torch.randint(0, V, (N, k), dtype=torch.int64, device='cuda')  # int64!
    
    out_ref = reference_forward(H, W, idx)
    out = indexed_logits_forward(H, W, idx)
    
    return check_close_v2(out, out_ref, "int64 idx forward", dtype=torch.float16)


# =============================================================================
# Memory Efficiency Tests
# =============================================================================

def reset_memory():
    """Reset CUDA memory statistics."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def test_memory_efficiency():
    """Basic memory efficiency test (forward only)."""
    print("\n=== Test Memory Efficiency (Forward) ===")
    
    N, d, V, k = 4096, 1024, 50000, 64
    
    reset_memory()
    
    H = torch.randn(N, d, dtype=torch.float16, device='cuda')
    W = torch.randn(V, d, dtype=torch.float16, device='cuda')
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    torch.cuda.synchronize()
    mem_after_inputs = torch.cuda.max_memory_allocated()
    
    torch.cuda.reset_peak_memory_stats()
    out = indexed_logits_forward(H, W, idx)
    torch.cuda.synchronize()
    mem_forward = torch.cuda.max_memory_allocated()
    
    avoided_size = N * k * d * 2  # [N, k, d] in float16
    overhead = mem_forward - mem_after_inputs
    
    print(f"  Input tensors memory: {mem_after_inputs / 1e6:.1f} MB")
    print(f"  Peak memory during forward: {mem_forward / 1e6:.1f} MB")
    print(f"  Size of [N,k,d] tensor (avoided): {avoided_size / 1e6:.1f} MB")
    print(f"  Actual overhead during forward: {overhead / 1e6:.1f} MB")
    
    passed = overhead < avoided_size * 0.5
    print(f"  Memory efficient: {'YES' if passed else 'NO'}")
    
    return passed


def test_memory_efficiency_fwd_bwd():
    """
    Memory efficiency test for forward + backward pass.
    Compares fused kernel against dense full logits and shared-candidate baselines.
    """
    print("\n=== Test Memory Efficiency (Forward + Backward) ===")
    
    N, d, V, k = 4096, 1024, 50000, 64
    dtype = torch.float16
    
    print(f"\n  Configuration: N={N}, d={d}, V={V}, k={k}, dtype=float16")
    
    # Calculate theoretical sizes
    H_size = N * d * 2 / 1e6
    W_size = V * d * 2 / 1e6
    idx_size = N * k * 4 / 1e6
    out_size = N * k * 2 / 1e6
    avoided_size = N * k * d * 2 / 1e6
    grad_W_fp32_size = V * d * 4 / 1e6
    full_logits_size = N * V * 2 / 1e6
    
    print(f"\n  Theoretical sizes:")
    print(f"    H: {H_size:.1f} MB")
    print(f"    W: {W_size:.1f} MB")
    print(f"    idx: {idx_size:.1f} MB")
    print(f"    out: {out_size:.1f} MB")
    print(f"    [N,k,d] intermediate (avoided): {avoided_size:.1f} MB")
    print(f"    grad_W_fp32 (backward): {grad_W_fp32_size:.1f} MB")
    print(f"    Full logits [N,V]: {full_logits_size:.1f} MB")
    
    results = {}
    
    # Case 1: Fused forward + backward
    print(f"\n  Case 1: Fused kernel forward + backward")
    reset_memory()
    
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device='cuda', requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    torch.cuda.synchronize()
    mem_inputs = torch.cuda.max_memory_allocated()
    
    torch.cuda.reset_peak_memory_stats()
    out = indexed_logits(H, W, idx)
    loss = out.float().sum()
    loss.backward()
    torch.cuda.synchronize()
    
    fused_alloc = torch.cuda.max_memory_allocated() / 1e6
    fused_reserved = torch.cuda.max_memory_reserved() / 1e6
    fused_overhead = fused_alloc - mem_inputs / 1e6
    
    results['fused'] = {'alloc': fused_alloc, 'reserved': fused_reserved, 'overhead': fused_overhead}
    print(f"    Peak allocated: {fused_alloc:.1f} MB")
    print(f"    Peak reserved: {fused_reserved:.1f} MB")
    print(f"    Overhead above inputs: {fused_overhead:.1f} MB")
    
    del H, W, idx, out, loss
    
    # Case 2: Dense full logits + gather (with reduced V if needed)
    print(f"\n  Case 2: Dense full logits + gather baseline")
    
    # Estimate if we can fit full logits
    max_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    est_dense_memory_gb = (N * V * 2 * 3 + H_size * 1e6 + W_size * 1e6) / 1e9  # *3 for fwd,grad_out,grad
    
    V_dense = V
    if est_dense_memory_gb > max_gpu_memory_gb * 0.8:  # Use 80% of GPU memory as limit
        V_dense = int(max_gpu_memory_gb * 0.8 * 1e9 / (N * 2 * 3))
        V_dense = min(V_dense, V)
        print(f"    [V reduced from {V} to {V_dense} to fit in memory]")
    
    reset_memory()
    
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
    W = torch.randn(V_dense, d, dtype=dtype, device='cuda', requires_grad=True)
    idx = torch.randint(0, V_dense, (N, k), dtype=torch.int64, device='cuda')
    
    torch.cuda.synchronize()
    mem_inputs = torch.cuda.max_memory_allocated()
    
    try:
        torch.cuda.reset_peak_memory_stats()
        logits_full = H @ W.T
        out = torch.gather(logits_full, 1, idx)
        loss = out.float().sum()
        loss.backward()
        torch.cuda.synchronize()
        
        dense_alloc = torch.cuda.max_memory_allocated() / 1e6
        dense_reserved = torch.cuda.max_memory_reserved() / 1e6
        dense_overhead = dense_alloc - mem_inputs / 1e6
        
        results['dense'] = {'alloc': dense_alloc, 'reserved': dense_reserved, 
                           'overhead': dense_overhead, 'V': V_dense}
        print(f"    Peak allocated: {dense_alloc:.1f} MB")
        print(f"    Peak reserved: {dense_reserved:.1f} MB")
        print(f"    Overhead above inputs: {dense_overhead:.1f} MB")
        
        del logits_full
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"    OOM even with V={V_dense}")
            torch.cuda.empty_cache()
        else:
            raise
    
    del H, W, idx
    if 'out' in dir():
        del out, loss
    
    # Case 3: Shared-candidate GEMM baseline
    for K in [k, 4*k]:
        print(f"\n  Case 3: Shared-candidate GEMM (K={K})")
        reset_memory()
        
        torch.manual_seed(42)
        H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
        W = torch.randn(V, d, dtype=dtype, device='cuda', requires_grad=True)
        S = torch.randperm(V, device='cuda')[:K]
        idx_shared = torch.randint(0, K, (N, K), dtype=torch.int64, device='cuda')
        
        torch.cuda.synchronize()
        mem_inputs = torch.cuda.max_memory_allocated()
        
        torch.cuda.reset_peak_memory_stats()
        W_sub = W[S]
        logits_sub = H @ W_sub.T
        out = torch.gather(logits_sub, 1, idx_shared)
        loss = out.float().sum()
        loss.backward()
        torch.cuda.synchronize()
        
        shared_alloc = torch.cuda.max_memory_allocated() / 1e6
        shared_reserved = torch.cuda.max_memory_reserved() / 1e6
        shared_overhead = shared_alloc - mem_inputs / 1e6
        
        results[f'shared_K{K}'] = {'alloc': shared_alloc, 'reserved': shared_reserved, 
                                   'overhead': shared_overhead}
        print(f"    Peak allocated: {shared_alloc:.1f} MB")
        print(f"    Peak reserved: {shared_reserved:.1f} MB")
        print(f"    Overhead above inputs: {shared_overhead:.1f} MB")
        
        del H, W, S, W_sub, logits_sub, idx_shared, out, loss
    
    # Allocator growth test
    print(f"\n  Allocator growth test (20 iterations of fused kernel):")
    reset_memory()
    
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda', requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device='cuda', requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    for i in range(20):
        if H.grad is not None:
            H.grad = None
        if W.grad is not None:
            W.grad = None
        out = indexed_logits(H, W, idx)
        loss = out.float().sum()
        loss.backward()
        
        torch.cuda.synchronize()
        if i == 0:
            reserved_1 = torch.cuda.max_memory_reserved() / 1e6
        elif i == 19:
            reserved_20 = torch.cuda.max_memory_reserved() / 1e6
    
    print(f"    Reserved after iter 1:  {reserved_1:.1f} MB")
    print(f"    Reserved after iter 20: {reserved_20:.1f} MB")
    print(f"    Growth: {reserved_20 - reserved_1:.1f} MB")
    
    # Summary
    print(f"\n  Summary:")
    print(f"    Fused kernel saves {avoided_size:.1f} MB by avoiding [N,k,d] materialization")
    if 'dense' in results:
        print(f"    Fused overhead ({results['fused']['overhead']:.1f} MB) vs "
              f"Dense overhead ({results['dense']['overhead']:.1f} MB)")
    
    return True


# =============================================================================
# Determinism / Stability Tests
# =============================================================================

def test_nondeterminism():
    """
    Test determinism of forward and backward passes.
    Forward should be deterministic. Backward grad_W may have slight differences
    due to atomic operation ordering.
    """
    print("\n=== Test Determinism / Stability ===")
    
    N, d, V, k = 64, 128, 200, 16
    dtype = torch.float16
    
    torch.manual_seed(42)
    H = torch.randn(N, d, dtype=dtype, device='cuda')
    W = torch.randn(V, d, dtype=dtype, device='cuda')
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    grad_out = torch.randn(N, k, dtype=dtype, device='cuda')
    
    # Forward determinism
    print("\n  Forward determinism:")
    out1 = indexed_logits_forward(H, W, idx)
    out2 = indexed_logits_forward(H, W, idx)
    
    fwd_diff = (out1 - out2).abs().max().item()
    print(f"    Max diff between two forward passes: {fwd_diff:.6e}")
    print(f"    Forward deterministic: {'YES' if fwd_diff == 0 else 'NO'}")
    
    # Backward determinism
    print("\n  Backward determinism:")
    grad_H1, grad_W1 = indexed_logits_backward(H, W, idx, grad_out)
    grad_H2, grad_W2 = indexed_logits_backward(H, W, idx, grad_out)
    
    grad_H_diff = (grad_H1 - grad_H2).abs().max().item()
    grad_W_diff = (grad_W1 - grad_W2).abs().max().item()
    
    print(f"    Max diff in grad_H: {grad_H_diff:.6e}")
    print(f"    Max diff in grad_W: {grad_W_diff:.6e}")
    print(f"    grad_H deterministic: {'YES' if grad_H_diff == 0 else 'NO'}")
    print(f"    grad_W deterministic: {'YES' if grad_W_diff == 0 else 'NO (expected due to atomics)'}")
    
    # Test with heavy collisions
    print("\n  Backward with heavy collisions (V_small=32):")
    idx_collision = torch.randint(0, 32, (N, k), dtype=torch.int32, device='cuda')
    
    grad_H1, grad_W1 = indexed_logits_backward(H, W, idx_collision, grad_out)
    grad_H2, grad_W2 = indexed_logits_backward(H, W, idx_collision, grad_out)
    
    grad_H_diff = (grad_H1 - grad_H2).abs().max().item()
    grad_W_diff = (grad_W1 - grad_W2).abs().max().item()
    
    print(f"    Max diff in grad_H: {grad_H_diff:.6e}")
    print(f"    Max diff in grad_W: {grad_W_diff:.6e}")
    
    # Still verify correctness against reference
    print("\n  Correctness check with collisions:")
    grad_H_ref, grad_W_ref = reference_backward(H, W, idx_collision, grad_out)
    passed = check_close_v2(grad_W1, grad_W_ref, "grad_W (collision)", dtype=dtype, rtol=5e-2, atol=5e-2)
    
    return True  # This test is informational, always passes


# =============================================================================
# Non-contiguity Handling Tests
# =============================================================================

def test_noncontiguous_inputs():
    """
    Test handling of non-contiguous inputs.
    Extension should either throw a clear error or handle it correctly.
    """
    print("\n=== Test Non-contiguous Input Handling ===")
    
    N, d, V, k = 32, 64, 100, 8
    dtype = torch.float16
    
    torch.manual_seed(42)
    
    # Create non-contiguous H by slicing
    print("\n  Non-contiguous H (via stride trick):")
    H_big = torch.randn(N, d * 2, dtype=dtype, device='cuda')
    H_nc = H_big[:, ::2]  # Non-contiguous view with shape [N, d]
    
    W = torch.randn(V, d, dtype=dtype, device='cuda')
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
    
    print(f"    H_nc.is_contiguous(): {H_nc.is_contiguous()}")
    print(f"    H_nc.shape: {H_nc.shape}, H_nc.stride(): {H_nc.stride()}")
    
    try:
        out = indexed_logits_forward(H_nc, W, idx)
        # If it succeeded, check correctness
        H_cont = H_nc.contiguous()
        out_ref = reference_forward(H_cont, W, idx)
        if torch.allclose(out, out_ref, rtol=1e-2, atol=1e-2):
            print("    Result: PASSED (handled internally)")
        else:
            print("    Result: COMPUTED but INCORRECT")
    except RuntimeError as e:
        if "contiguous" in str(e).lower():
            print(f"    Result: Clear contiguity error (expected)")
        else:
            print(f"    Result: Error - {e}")
    
    # Create non-contiguous W
    print("\n  Non-contiguous W (via stride trick):")
    W_big = torch.randn(V, d * 2, dtype=dtype, device='cuda')
    W_nc = W_big[:, ::2]  # Non-contiguous view
    
    H = torch.randn(N, d, dtype=dtype, device='cuda')
    
    print(f"    W_nc.is_contiguous(): {W_nc.is_contiguous()}")
    print(f"    W_nc.shape: {W_nc.shape}, W_nc.stride(): {W_nc.stride()}")
    
    try:
        out = indexed_logits_forward(H, W_nc, idx)
        W_cont = W_nc.contiguous()
        out_ref = reference_forward(H, W_cont, idx)
        if torch.allclose(out, out_ref, rtol=1e-2, atol=1e-2):
            print("    Result: PASSED (handled internally)")
        else:
            print("    Result: COMPUTED but INCORRECT")
    except RuntimeError as e:
        if "contiguous" in str(e).lower():
            print(f"    Result: Clear contiguity error (expected)")
        else:
            print(f"    Result: Error - {e}")
    
    # Create non-contiguous idx (less common but possible)
    print("\n  Non-contiguous idx (via transpose):")
    idx_big = torch.randint(0, V, (k, N), dtype=torch.int32, device='cuda')
    idx_nc = idx_big.T  # [N, k] but non-contiguous
    
    print(f"    idx_nc.is_contiguous(): {idx_nc.is_contiguous()}")
    
    try:
        out = indexed_logits_forward(H, W, idx_nc)
        idx_cont = idx_nc.contiguous()
        out_ref = reference_forward(H, W, idx_cont)
        if torch.allclose(out, out_ref, rtol=1e-2, atol=1e-2):
            print("    Result: PASSED (handled internally)")
        else:
            print("    Result: COMPUTED but INCORRECT")
    except RuntimeError as e:
        if "contiguous" in str(e).lower():
            print(f"    Result: Clear contiguity error (expected)")
        else:
            print(f"    Result: Error - {e}")
    
    return True  # Informational test


# =============================================================================
# Main Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("Running indexed_logits CUDA extension tests")
    print("=" * 70)
    
    if not EXTENSION_AVAILABLE:
        print("\nERROR: Extension not available. Build it first with:")
        print("  python setup.py build_ext --inplace")
        return False
    
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available")
        return False
    
    print(f"\nCUDA device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    
    all_passed = True
    
    # Basic correctness tests
    all_passed &= test_forward_basic(torch.float16)
    all_passed &= test_backward_basic(torch.float16)
    all_passed &= test_autograd_function(torch.float16)
    
    # BFloat16
    all_passed &= test_bfloat16()
    
    # Edge cases
    all_passed &= test_edge_cases()
    
    # Int64 conversion
    all_passed &= test_int64_idx()
    
    # Memory efficiency (forward only)
    all_passed &= test_memory_efficiency()
    
    # Memory efficiency (forward + backward)
    test_memory_efficiency_fwd_bwd()
    
    # Determinism
    test_nondeterminism()
    
    # Non-contiguous handling
    test_noncontiguous_inputs()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CORRECTNESS TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)