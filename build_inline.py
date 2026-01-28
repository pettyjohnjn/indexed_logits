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

import torch
from torch.utils.cpp_extension import load_inline
import os

# C++ source code (indexed_logits.cpp content)

cpp_source = """
#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor indexed_logits_forward_cuda(
    torch::Tensor H,
    torch::Tensor W,
    torch::Tensor idx
);

std::vector<torch::Tensor> indexed_logits_backward_cuda(
    torch::Tensor H,
    torch::Tensor W,
    torch::Tensor idx,
    torch::Tensor grad_out
);

// Validation helper
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor indexed_logits_forward(
    torch::Tensor H,
    torch::Tensor W,
    torch::Tensor idx
) {
    CHECK_INPUT(H);
    CHECK_INPUT(W);
    CHECK_INPUT(idx);
    
    TORCH_CHECK(
        H.scalar_type() == torch::kFloat16 || H.scalar_type() == torch::kBFloat16,
        "H must be float16 or bfloat16, got ", H.scalar_type()
    );
    TORCH_CHECK(W.scalar_type() == H.scalar_type(), "W must have same dtype as H");
    
    if (idx.scalar_type() == torch::kInt64) {
        TORCH_WARN("idx is int64, converting to int32.");
        idx = idx.to(torch::kInt32);
    }
    TORCH_CHECK(idx.scalar_type() == torch::kInt32, "idx must be int32");
    
    TORCH_CHECK(H.dim() == 2, "H must be 2D [N, d]");
    TORCH_CHECK(W.dim() == 2, "W must be 2D [V, d]");
    TORCH_CHECK(idx.dim() == 2, "idx must be 2D [N, k]");
    TORCH_CHECK(W.size(1) == H.size(1), "W.size(1) must equal H.size(1)");
    TORCH_CHECK(idx.size(0) == H.size(0), "idx.size(0) must equal H.size(0)");
    
    return indexed_logits_forward_cuda(H, W, idx);
}

std::vector<torch::Tensor> indexed_logits_backward(
    torch::Tensor H,
    torch::Tensor W,
    torch::Tensor idx,
    torch::Tensor grad_out
) {
    CHECK_INPUT(H);
    CHECK_INPUT(W);
    CHECK_INPUT(idx);
    CHECK_INPUT(grad_out);
    
    TORCH_CHECK(
        H.scalar_type() == torch::kFloat16 || H.scalar_type() == torch::kBFloat16,
        "H must be float16 or bfloat16"
    );
    TORCH_CHECK(W.scalar_type() == H.scalar_type(), "W dtype mismatch");
    TORCH_CHECK(grad_out.scalar_type() == H.scalar_type(), "grad_out dtype mismatch");
    
    if (idx.scalar_type() == torch::kInt64) {
        idx = idx.to(torch::kInt32);
    }
    TORCH_CHECK(idx.scalar_type() == torch::kInt32, "idx must be int32");
    
    return indexed_logits_backward_cuda(H, W, idx, grad_out);
}
"""

# CUDA source code (indexed_logits_cuda.cu content)

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CDIV(x, y) (((x) + (y) - 1) / (y))

// Forward kernel
template <typename scalar_t>
__global__ void indexed_logits_forward_kernel(
    const scalar_t* __restrict__ H,
    const scalar_t* __restrict__ W,
    const int32_t* __restrict__ idx,
    scalar_t* __restrict__ out,
    const int64_t N,
    const int64_t d,
    const int64_t k
) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = N * k;
    
    if (global_idx >= total_elements) return;
    
    const int64_t i = global_idx / k;
    const int64_t j = global_idx % k;
    const int32_t token = idx[i * k + j];
    
    float acc = 0.0f;
    for (int64_t t = 0; t < d; t++) {
        float h_val = static_cast<float>(H[i * d + t]);
        float w_val = static_cast<float>(W[token * d + t]);
        acc += h_val * w_val;
    }
    
    out[i * k + j] = static_cast<scalar_t>(acc);
}

// grad_H kernel
template <typename scalar_t>
__global__ void indexed_logits_backward_grad_H_kernel(
    const scalar_t* __restrict__ W,
    const int32_t* __restrict__ idx,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_H,
    const int64_t N,
    const int64_t d,
    const int64_t k
) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = N * d;
    
    if (global_idx >= total_elements) return;
    
    const int64_t i = global_idx / d;
    const int64_t t = global_idx % d;
    
    float acc = 0.0f;
    for (int64_t j = 0; j < k; j++) {
        const int32_t token = idx[i * k + j];
        float g = static_cast<float>(grad_out[i * k + j]);
        float w = static_cast<float>(W[token * d + t]);
        acc += g * w;
    }
    
    grad_H[i * d + t] = static_cast<scalar_t>(acc);
}

// grad_W kernel
template <typename scalar_t>
__global__ void indexed_logits_backward_grad_W_kernel(
    const scalar_t* __restrict__ H,
    const int32_t* __restrict__ idx,
    const scalar_t* __restrict__ grad_out,
    float* __restrict__ grad_W_fp32,
    const int64_t N,
    const int64_t d,
    const int64_t k
) {
    const int64_t pair_idx = blockIdx.x;
    const int64_t d_block = blockIdx.y;
    const int64_t t_local = threadIdx.x;
    
    const int64_t total_pairs = N * k;
    if (pair_idx >= total_pairs) return;
    
    const int64_t i = pair_idx / k;
    const int64_t j = pair_idx % k;
    
    const int32_t token = idx[i * k + j];
    const float g = static_cast<float>(grad_out[i * k + j]);
    
    const int64_t t = d_block * blockDim.x + t_local;
    if (t < d) {
        float h = static_cast<float>(H[i * d + t]);
        atomicAdd(&grad_W_fp32[token * d + t], g * h);
    }
}

// Launch helpers
torch::Tensor indexed_logits_forward_cuda(
    torch::Tensor H,
    torch::Tensor W,
    torch::Tensor idx
) {
    const int64_t N = H.size(0);
    const int64_t d = H.size(1);
    const int64_t k = idx.size(1);
    
    auto out = torch::empty({N, k}, H.options());
    
    const int64_t total_elements = N * k;
    const int threads = 256;
    const int blocks = CDIV(total_elements, threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        H.scalar_type(), "indexed_logits_forward_cuda", ([&] {
            indexed_logits_forward_kernel<scalar_t><<<blocks, threads>>>(
                H.data_ptr<scalar_t>(),
                W.data_ptr<scalar_t>(),
                idx.data_ptr<int32_t>(),
                out.data_ptr<scalar_t>(),
                N, d, k
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    
    return out;
}

std::vector<torch::Tensor> indexed_logits_backward_cuda(
    torch::Tensor H,
    torch::Tensor W,
    torch::Tensor idx,
    torch::Tensor grad_out
) {
    const int64_t N = H.size(0);
    const int64_t d = H.size(1);
    const int64_t V = W.size(0);
    const int64_t k = idx.size(1);
    
    auto grad_H = torch::empty({N, d}, H.options());
    auto grad_W_fp32 = torch::zeros({V, d}, torch::dtype(torch::kFloat32).device(H.device()));
    
    const int threads = 256;
    
    // grad_H
    {
        const int64_t total = N * d;
        const int blocks = CDIV(total, threads);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            H.scalar_type(), "grad_H", ([&] {
                indexed_logits_backward_grad_H_kernel<scalar_t><<<blocks, threads>>>(
                    W.data_ptr<scalar_t>(),
                    idx.data_ptr<int32_t>(),
                    grad_out.data_ptr<scalar_t>(),
                    grad_H.data_ptr<scalar_t>(),
                    N, d, k
                );
            })
        );
    }
    
    // grad_W
    {
        const int64_t total_pairs = N * k;
        const int d_blocks = CDIV(d, threads);
        dim3 grid(total_pairs, d_blocks);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            H.scalar_type(), "grad_W", ([&] {
                indexed_logits_backward_grad_W_kernel<scalar_t><<<grid, threads>>>(
                    H.data_ptr<scalar_t>(),
                    idx.data_ptr<int32_t>(),
                    grad_out.data_ptr<scalar_t>(),
                    grad_W_fp32.data_ptr<float>(),
                    N, d, k
                );
            })
        );
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    
    return {grad_H, grad_W_fp32};
}
"""

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
