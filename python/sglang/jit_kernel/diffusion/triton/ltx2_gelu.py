# Adapted from NVlabs/Sana sol-engine LTX2 GELU fusion.

import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_bias_gelu_tanh_inplace_kernel(
    x_ptr,
    bias_ptr,
    n_elements: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    cols = offsets % hidden
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + bias
    x3 = x * x * x
    y = x * tl.sigmoid(1.5957691216057308 * (x + 0.044715 * x3))
    tl.store(x_ptr + offsets, y, mask=mask)


def ltx2_bias_gelu_tanh_inplace(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must be a CUDA fp16/bf16 tensor")
    if not bias.is_cuda or bias.dtype != x.dtype:
        raise ValueError("bias must be a CUDA tensor with the same dtype as x")
    if not x.is_contiguous() or not bias.is_contiguous():
        raise ValueError("x and bias must be contiguous")
    if x.ndim < 1 or bias.ndim != 1 or x.shape[-1] != bias.shape[0]:
        raise ValueError("bias must match x last dimension")
    n_elements = x.numel()
    if n_elements == 0:
        return x
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _ltx2_bias_gelu_tanh_inplace_kernel[grid](
        x,
        bias,
        n_elements,
        int(x.shape[-1]),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return x
