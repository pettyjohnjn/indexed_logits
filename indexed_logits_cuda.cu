// indexed_logits_cuda.cu
// CUDA kernels for indexed logits computation
// Computes out[i,j] = dot(H[i,:], W[idx[i,j],:]) without materializing W[idx]

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Helper for ceiling division
#define CDIV(x, y) (((x) + (y) - 1) / (y))

// Forward Kernel
// Each thread computes one output element out[i,j] = dot(H[i,:], W[idx[i,j],:])

template <typename scalar_t>
__global__ void indexed_logits_forward_kernel(
    const scalar_t* __restrict__ H,      // [N, d]
    const scalar_t* __restrict__ W,      // [V, d]
    const int32_t* __restrict__ idx,     // [N, k]
    scalar_t* __restrict__ out,          // [N, k]
    const int64_t N,
    const int64_t d,
    const int64_t k
) {
    // Global thread index maps to (i, j) in output
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = N * k;
    
    if (global_idx >= total_elements) return;
    
    const int64_t i = global_idx / k;  // row in H
    const int64_t j = global_idx % k;  // which of the k indices
    
    // Get the token index for this position
    const int32_t token = idx[i * k + j];
    
    // Compute dot product in float32 for numerical stability
    float acc = 0.0f;
    for (int64_t t = 0; t < d; t++) {
        float h_val = static_cast<float>(H[i * d + t]);
        float w_val = static_cast<float>(W[token * d + t]);
        acc += h_val * w_val;
    }
    
    // Write result, casting back to output dtype
    out[i * k + j] = static_cast<scalar_t>(acc);
}

// Backward Kernel A: grad_H
// grad_H[i, t] = sum_j grad_out[i, j] * W[idx[i,j], t]
// Grid over (i, t_chunk) where each thread handles a chunk of dimensions

template <typename scalar_t>
__global__ void indexed_logits_backward_grad_H_kernel(
    const scalar_t* __restrict__ W,           // [V, d]
    const int32_t* __restrict__ idx,          // [N, k]
    const scalar_t* __restrict__ grad_out,    // [N, k]
    scalar_t* __restrict__ grad_H,            // [N, d]
    const int64_t N,
    const int64_t d,
    const int64_t k
) {
    // Each thread handles one (i, t) position
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = N * d;
    
    if (global_idx >= total_elements) return;
    
    const int64_t i = global_idx / d;  // position index
    const int64_t t = global_idx % d;  // dimension index
    
    // Accumulate over all k indices
    float acc = 0.0f;
    for (int64_t j = 0; j < k; j++) {
        const int32_t token = idx[i * k + j];
        float g = static_cast<float>(grad_out[i * k + j]);
        float w = static_cast<float>(W[token * d + t]);
        acc += g * w;
    }
    
    grad_H[i * d + t] = static_cast<scalar_t>(acc);
}

// Backward Kernel B: grad_W
// grad_W[token, t] += sum_{(i,j): idx[i,j]=token} grad_out[i,j] * H[i, t]
// Uses atomic adds in float32 to handle collisions

template <typename scalar_t>
__global__ void indexed_logits_backward_grad_W_kernel(
    const scalar_t* __restrict__ H,           // [N, d]
    const int32_t* __restrict__ idx,          // [N, k]
    const scalar_t* __restrict__ grad_out,    // [N, k]
    float* __restrict__ grad_W_fp32,          // [V, d] in float32
    const int64_t N,
    const int64_t d,
    const int64_t k
) {
    // Grid over (i, j) pairs - each thread handles one (i, j) and loops over d
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_pairs = N * k;
    
    if (global_idx >= total_pairs) return;
    
    const int64_t i = global_idx / k;
    const int64_t j = global_idx % k;
    
    const int32_t token = idx[i * k + j];
    const float g = static_cast<float>(grad_out[i * k + j]);
    
    // For each dimension, atomically add contribution to grad_W
    for (int64_t t = 0; t < d; t++) {
        float h = static_cast<float>(H[i * d + t]);
        atomicAdd(&grad_W_fp32[token * d + t], g * h);
    }
}

// Optimized grad_W kernel with shared memory reduction for the grad_out value
// This processes multiple dimensions per thread to reduce atomic contention

template <typename scalar_t, int DIMS_PER_THREAD>
__global__ void indexed_logits_backward_grad_W_kernel_v2(
    const scalar_t* __restrict__ H,           // [N, d]
    const int32_t* __restrict__ idx,          // [N, k]
    const scalar_t* __restrict__ grad_out,    // [N, k]
    float* __restrict__ grad_W_fp32,          // [V, d] in float32
    const int64_t N,
    const int64_t d,
    const int64_t k
) {
    // Each block handles multiple (i, j, t_start) combinations
    const int64_t pair_idx = blockIdx.x;
    const int64_t d_block = blockIdx.y;  // which chunk of dimensions
    const int64_t t_local = threadIdx.x;
    
    const int64_t total_pairs = N * k;
    if (pair_idx >= total_pairs) return;
    
    const int64_t i = pair_idx / k;
    const int64_t j = pair_idx % k;
    
    const int32_t token = idx[i * k + j];
    const float g = static_cast<float>(grad_out[i * k + j]);
    
    // Each thread handles one dimension
    const int64_t t = d_block * blockDim.x + t_local;
    if (t < d) {
        float h = static_cast<float>(H[i * d + t]);
        atomicAdd(&grad_W_fp32[token * d + t], g * h);
    }
}

// Launch helpers with dtype dispatch

torch::Tensor indexed_logits_forward_cuda(
    torch::Tensor H,
    torch::Tensor W,
    torch::Tensor idx
) {
    const int64_t N = H.size(0);
    const int64_t d = H.size(1);
    const int64_t k = idx.size(1);
    
    // Allocate output with same dtype as H
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
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in forward kernel: ", cudaGetErrorString(err));
    
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
    
    // Allocate grad_H with same dtype as H
    auto grad_H = torch::empty({N, d}, H.options());
    
    // Allocate grad_W in float32 for atomic accumulation
    // This avoids fp16/bf16 atomic issues and ensures numerical stability
    auto grad_W_fp32 = torch::zeros({V, d}, torch::dtype(torch::kFloat32).device(H.device()));
    
    const int threads = 256;
    
    // Launch grad_H kernel
    {
        const int64_t total = N * d;
        const int blocks = CDIV(total, threads);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            H.scalar_type(), "indexed_logits_backward_grad_H", ([&] {
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
    
    // Launch grad_W kernel
    // Using version 2 with grid over (N*k, ceil(d/threads))
    {
        const int64_t total_pairs = N * k;
        const int d_blocks = CDIV(d, threads);
        dim3 grid(total_pairs, d_blocks);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            H.scalar_type(), "indexed_logits_backward_grad_W", ([&] {
                indexed_logits_backward_grad_W_kernel_v2<scalar_t, 1><<<grid, threads>>>(
                    H.data_ptr<scalar_t>(),
                    idx.data_ptr<int32_t>(),
                    grad_out.data_ptr<scalar_t>(),
                    grad_W_fp32.data_ptr<float>(),
                    N, d, k
                );
            })
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in backward kernel: ", cudaGetErrorString(err));
    
    // Note: We return grad_W in float32 intentionally.
    // This is acceptable for autograd and provides better numerical stability.
    // The optimizer will handle the dtype conversion if needed.
    // If you need the same dtype, uncomment the following:
    // auto grad_W = grad_W_fp32.to(H.dtype());
    
    return {grad_H, grad_W_fp32};
}
