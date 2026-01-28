// indexed_logits.cpp
// PyBind11 bindings and dispatch for the indexed_logits CUDA extension

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
    // Device checks
    CHECK_INPUT(H);
    CHECK_INPUT(W);
    CHECK_INPUT(idx);
    
    // Dtype checks for H and W
    TORCH_CHECK(
        H.scalar_type() == torch::kFloat16 || H.scalar_type() == torch::kBFloat16,
        "H must be float16 or bfloat16, got ", H.scalar_type()
    );
    TORCH_CHECK(
        W.scalar_type() == H.scalar_type(),
        "W must have same dtype as H"
    );
    
    // idx dtype check - prefer int32, accept int64 with warning
    if (idx.scalar_type() == torch::kInt64) {
        TORCH_WARN("idx is int64, converting to int32. Consider using int32 directly for efficiency.");
        idx = idx.to(torch::kInt32);
    }
    TORCH_CHECK(
        idx.scalar_type() == torch::kInt32,
        "idx must be int32 (or int64 which will be converted), got ", idx.scalar_type()
    );
    
    // Shape checks
    TORCH_CHECK(H.dim() == 2, "H must be 2D [N, d]");
    TORCH_CHECK(W.dim() == 2, "W must be 2D [V, d]");
    TORCH_CHECK(idx.dim() == 2, "idx must be 2D [N, k]");
    
    int64_t N = H.size(0);
    int64_t d = H.size(1);
    int64_t V = W.size(0);
    int64_t k = idx.size(1);
    
    TORCH_CHECK(W.size(1) == d, "W.size(1) must equal H.size(1) (d)");
    TORCH_CHECK(idx.size(0) == N, "idx.size(0) must equal H.size(0) (N)");
    
    return indexed_logits_forward_cuda(H, W, idx);
}

std::vector<torch::Tensor> indexed_logits_backward(
    torch::Tensor H,
    torch::Tensor W,
    torch::Tensor idx,
    torch::Tensor grad_out
) {
    // Device checks
    CHECK_INPUT(H);
    CHECK_INPUT(W);
    CHECK_INPUT(idx);
    CHECK_INPUT(grad_out);
    
    // Dtype checks
    TORCH_CHECK(
        H.scalar_type() == torch::kFloat16 || H.scalar_type() == torch::kBFloat16,
        "H must be float16 or bfloat16"
    );
    TORCH_CHECK(W.scalar_type() == H.scalar_type(), "W must have same dtype as H");
    TORCH_CHECK(grad_out.scalar_type() == H.scalar_type(), "grad_out must have same dtype as H");
    
    // idx dtype
    if (idx.scalar_type() == torch::kInt64) {
        idx = idx.to(torch::kInt32);
    }
    TORCH_CHECK(idx.scalar_type() == torch::kInt32, "idx must be int32");
    
    // Shape checks
    TORCH_CHECK(H.dim() == 2 && W.dim() == 2 && idx.dim() == 2 && grad_out.dim() == 2,
                "All inputs must be 2D");
    
    int64_t N = H.size(0);
    int64_t d = H.size(1);
    int64_t V = W.size(0);
    int64_t k = idx.size(1);
    
    TORCH_CHECK(W.size(1) == d, "W dimension mismatch");
    TORCH_CHECK(idx.size(0) == N && idx.size(1) == k, "idx shape mismatch");
    TORCH_CHECK(grad_out.size(0) == N && grad_out.size(1) == k, "grad_out shape mismatch");
    
    return indexed_logits_backward_cuda(H, W, idx, grad_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &indexed_logits_forward, "Indexed logits forward (CUDA)");
    m.def("backward", &indexed_logits_backward, "Indexed logits backward (CUDA)");
}
