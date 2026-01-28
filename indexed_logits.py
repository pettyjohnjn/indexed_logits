"""
indexed_logits.py

Python wrapper for the indexed_logits CUDA extension.
Provides an autograd.Function for automatic differentiation.

Usage:
    from indexed_logits import indexed_logits
    
    # H: [N, d] hidden states (float16 or bfloat16)
    # W: [V, d] weight matrix (same dtype as H)
    # idx: [N, k] token indices (int32 preferred, int64 accepted)
    
    out = indexed_logits(H, W, idx)  # [N, k]
    
    # Supports backward pass:
    loss = out.sum()
    loss.backward()
"""

import torch
from torch.autograd import Function

# Import the compiled CUDA extension
# This will be available after running setup.py or using load_inline
try:
    import indexed_logits_cuda
except ImportError as e:
    raise ImportError(
        "Could not import indexed_logits_cuda. "
        "Please build the extension first using:\n"
        "  python setup.py install\n"
        "or\n"
        "  python setup.py build_ext --inplace\n"
        f"Original error: {e}"
    )


def indexed_logits_forward(H: torch.Tensor, W: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Compute indexed logits: out[i,j] = dot(H[i,:], W[idx[i,j],:])
    
    Args:
        H: Hidden states tensor of shape [N, d], dtype float16 or bfloat16
        W: Weight matrix of shape [V, d], same dtype as H
        idx: Index tensor of shape [N, k], dtype int32 (int64 accepted but converted)
    
    Returns:
        out: Logits tensor of shape [N, k], same dtype as H
    
    Note:
        This function does NOT allocate any [N, k, d] intermediate tensors.
        Peak memory is dominated by H, W, idx, and out.
    """
    return indexed_logits_cuda.forward(H, W, idx)


def indexed_logits_backward(
    H: torch.Tensor, 
    W: torch.Tensor, 
    idx: torch.Tensor, 
    grad_out: torch.Tensor
) -> tuple:
    """
    Compute gradients for indexed logits operation.
    
    Args:
        H: Hidden states tensor of shape [N, d]
        W: Weight matrix of shape [V, d]
        idx: Index tensor of shape [N, k]
        grad_out: Gradient of loss w.r.t. output, shape [N, k]
    
    Returns:
        grad_H: Gradient w.r.t. H, shape [N, d], same dtype as H
        grad_W: Gradient w.r.t. W, shape [V, d], dtype float32
    
    Note:
        grad_W is returned in float32 for numerical stability during atomic
        accumulation. PyTorch autograd accepts mixed dtypes in gradients.
        The optimizer will handle any necessary dtype conversion.
    """
    return indexed_logits_cuda.backward(H, W, idx, grad_out)


class IndexedLogits(Function):
    """
    Autograd Function for indexed logits computation.
    
    Computes out[i,j] = dot(H[i,:], W[idx[i,j],:]) for:
        - i in [0, N-1]
        - j in [0, k-1]
    
    This is memory-efficient: it does NOT materialize W[idx] as an [N, k, d] tensor.
    
    Forward:
        Accumulates dot products in float32, then casts to output dtype.
    
    Backward:
        - grad_H[i,:] = sum_j grad_out[i,j] * W[idx[i,j],:]
        - grad_W[token,:] += sum_{(i,j): idx[i,j]=token} grad_out[i,j] * H[i,:]
        
        grad_W uses float32 atomics for correctness and is returned in float32.
        grad_idx is None (indices are not differentiable).
    """
    
    @staticmethod
    def forward(ctx, H: torch.Tensor, W: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute indexed logits.
        
        Args:
            ctx: Context object for saving tensors
            H: [N, d] hidden states (float16 or bfloat16)
            W: [V, d] weight matrix (same dtype as H)
            idx: [N, k] token indices (int32)
        
        Returns:
            out: [N, k] logits (same dtype as H)
        """
        # Validate inputs
        if not H.is_cuda:
            raise RuntimeError("IndexedLogits only supports CUDA tensors. Got CPU tensor for H.")
        if not W.is_cuda:
            raise RuntimeError("IndexedLogits only supports CUDA tensors. Got CPU tensor for W.")
        if not idx.is_cuda:
            raise RuntimeError("IndexedLogits only supports CUDA tensors. Got CPU tensor for idx.")
        
        # Convert idx to int32 if needed
        if idx.dtype == torch.int64:
            idx = idx.to(torch.int32)
        
        # Save for backward
        ctx.save_for_backward(H, W, idx)
        
        return indexed_logits_cuda.forward(H, W, idx)
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Backward pass: compute gradients.
        
        Args:
            ctx: Context with saved tensors
            grad_out: [N, k] gradient of loss w.r.t. output
        
        Returns:
            grad_H: [N, d] gradient w.r.t. H (same dtype as H)
            grad_W: [V, d] gradient w.r.t. W (float32 for stability)
            None: grad_idx is None (not differentiable)
        """

        grad_out = grad_out.contiguous()
        
        H, W, idx = ctx.saved_tensors
        
        grad_H, grad_W = indexed_logits_cuda.backward(H, W, idx, grad_out)
        
        # grad_idx is None since indices are not differentiable
        return grad_H, grad_W, None


def indexed_logits(H: torch.Tensor, W: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Compute indexed logits with automatic differentiation support.
    
    This is the main entry point for the indexed logits operation.
    
    Computes: out[i,j] = dot(H[i,:], W[idx[i,j],:])
    
    Args:
        H: Hidden states tensor of shape [N, d]
           - N = B*T (flattened batch and sequence dimensions)
           - d = hidden dimension
           - dtype: float16 or bfloat16
           - Must be contiguous and on CUDA
        
        W: Weight/embedding matrix of shape [V, d]
           - V = vocabulary size
           - d = hidden dimension (must match H)
           - dtype: must match H
           - Must be contiguous and on CUDA
        
        idx: Token index tensor of shape [N, k]
           - k = number of top-k tokens per position
           - Each entry in [0, V-1]
           - dtype: int32 (int64 accepted but converted with warning)
           - Must be contiguous and on CUDA
    
    Returns:
        out: Logits tensor of shape [N, k]
           - Same dtype as H
           - out[i,j] = dot product of H[i,:] and W[idx[i,j],:]
    
    Memory efficiency:
        This function does NOT allocate any intermediate tensor of shape [N, k, d].
        Peak memory usage is:
        - H: N * d * sizeof(dtype)
        - W: V * d * sizeof(dtype)
        - idx: N * k * 4 bytes
        - out: N * k * sizeof(dtype)
        - (backward only) grad_W_fp32: V * d * 4 bytes
    
    Example:
        >>> import torch
        >>> from indexed_logits import indexed_logits
        >>> 
        >>> N, d, V, k = 1024, 256, 50000, 32
        >>> H = torch.randn(N, d, dtype=torch.float16, device='cuda')
        >>> W = torch.randn(V, d, dtype=torch.float16, device='cuda')
        >>> idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
        >>> 
        >>> out = indexed_logits(H, W, idx)
        >>> print(out.shape)  # torch.Size([1024, 32])
        >>> 
        >>> # With gradients:
        >>> H.requires_grad = True
        >>> W.requires_grad = True
        >>> out = indexed_logits(H, W, idx)
        >>> loss = out.sum()
        >>> loss.backward()
        >>> print(H.grad.shape)  # torch.Size([1024, 256])
        >>> print(W.grad.shape)  # torch.Size([50000, 256])
    """
    return IndexedLogits.apply(H, W, idx)
