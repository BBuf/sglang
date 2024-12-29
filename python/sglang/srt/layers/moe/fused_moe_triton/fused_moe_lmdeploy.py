# Modified from https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/kernels/cuda/fused_moe.py

import functools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from vllm import _custom_ops as ops
from vllm.logger import init_logger

from sglang.srt.layers.moe.topk import select_experts

logger = init_logger(__name__)


@triton.jit
def fused_moe_kernel(
    A,
    B,
    C,
    a_scale_ptr,
    b_scale_ptr,
    SortedIdx,
    ExpStart,
    ExpEnd,
    Weights,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M_NP2: tl.constexpr,
    ENABLE_WEIGHTS: tl.constexpr,
    top_k: tl.constexpr,
    expert_offset: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    compute_type: tl.constexpr,
):
    """fused moe kernel."""
    exp_id = tl.program_id(1)
    pid = tl.program_id(0)

    exp_start = tl.load(ExpStart + exp_id + expert_offset)
    exp_end = tl.load(ExpEnd + exp_id + expert_offset)
    M = exp_end - exp_start
    if M <= 0:
        return

    num_pid_m = tl.cdiv(M_NP2, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
        # pid_m = pid // num_pid_n
        # pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_sid = exp_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_sid = offs_sid < exp_end
    sid = tl.load(SortedIdx + offs_sid, mask=mask_sid, other=0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if reindex_a:
        offs_am = sid // top_k
    else:
        offs_am = offs_sid
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # deepseek has 160 experts, exp index would overflow int32
    exp_off = stride_be * exp_id.to(tl.int64)
    b_ptrs = B + exp_off + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if use_fp8_w8a8:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + exp_id)
    elif use_int8_w8a16:
        b_scale = tl.load(b_scale_ptr + offs_bn[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=mask_sid[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ENABLE_WEIGHTS:
        weight = tl.load(Weights + sid, mask=mask_sid)
        accumulator = accumulator * weight[:, None].to(accumulator.dtype)

    if use_int8_w8a16:
        c = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        c = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        c = accumulator.to(compute_type)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = offs_sid
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=mask_sid[:, None])


def fused_moe_kernel_launcher(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
    weights: torch.Tensor,
    enable_weights: bool,
    top_k: int,
    num_tokens: int,
    expert_offset: int,
    reindex_a: bool,
    reindex_c: bool,
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    config: Dict[str, Any],
):
    """fused moe kernel launcher."""

    if num_tokens is None:
        num_tokens = A.size(0)
    M_NP2 = triton.next_power_of_2(num_tokens)
    M_NP2 = max(64, M_NP2)
    E, N, K = B.shape

    def _grid_fn(META):
        grid = (
            triton.cdiv(M_NP2, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            E,
        )
        return grid

    A = A.flatten(0, -2)
    C = C.flatten(0, -2)
    if not use_fp8_w8a8 and not use_int8_w8a16:
        assert A_scale is None
        assert B_scale is None
    else:
        if use_fp8_w8a8:
            A, A_scale = ops.scaled_fp8_quant(A, A_scale)
        assert B_scale is not None

    grid = _grid_fn
    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        sorted_idx,
        exp_start,
        exp_end,
        weights,
        N=N,
        K=K,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_be=B.stride(0),
        stride_bn=B.stride(1),
        stride_bk=B.stride(2),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        ENABLE_WEIGHTS=enable_weights,
        top_k=top_k,
        expert_offset=expert_offset,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        M_NP2=M_NP2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        compute_type=compute_type,
        **config,
    )


@triton.jit
def _start_end_kernel(
    TopkIdx,
    SortedIdx,
    ExpStart,
    ExpEnd,
    len_sorted_idx: int,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """start end kernel."""
    exp_id = tl.program_id(0)
    exp_start = -1
    cnt = 0

    s_off = tl.arange(0, BLOCK)

    # find start
    for sidx_start in range(0, len_sorted_idx, BLOCK):
        sidx_off = sidx_start + s_off
        sidx_mask = sidx_off < len_sorted_idx
        sidx = tl.load(SortedIdx + sidx_off, mask=sidx_mask, other=0)
        tidx = tl.load(TopkIdx + sidx, mask=sidx_mask, other=num_experts)
        tidx_mask = tidx == exp_id
        cnt += tl.sum(tidx_mask.to(tl.int32))
        if cnt > 0 and exp_start < 0:
            exp_start = sidx_start + tl.argmax(tidx_mask, axis=0)

    if exp_start < 0:
        exp_start *= 0
    exp_end = exp_start + cnt
    tl.store(ExpStart + exp_id, exp_start)
    tl.store(ExpEnd + exp_id, exp_end)


def get_start_end(topk_idx: torch.Tensor, sorted_idx: torch.Tensor, num_experts: int):
    """get start and end.

    same process as:
    >>> exp_tok_cnt = F.one_hot(flatten_topk_ids, num_classes=E).sum(0)
    >>> exp_end = exp_tok_cnt.cumsum(0)
    >>> exp_start = exp_end - exp_tok_cnt
    """
    start_end = sorted_idx.new_empty(2, num_experts)
    exp_start = start_end[0, :]
    exp_end = start_end[1, :]

    BLOCK = 128
    _start_end_kernel[(num_experts,)](
        topk_idx,
        sorted_idx,
        exp_start,
        exp_end,
        len_sorted_idx=sorted_idx.numel(),
        num_experts=num_experts,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=1,
    )

    return exp_start, exp_end


def get_config_file_name(E: int, N: int, dtype: Optional[str]) -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}_lmdeploy.json"


@functools.lru_cache
def get_moe_configs(E: int, N: int, dtype: Optional[str]) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N, dtype)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs/lmdeploy", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info("Using configuration from %s for MoE layer.", config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default MoE config. Performance might be sub-optimal! "
            "Config file not found at %s"
        ),
        config_file_path,
    )
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
) -> Dict[str, int]:
    if dtype == "fp8_w8a8":
        config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 8,
            "num_stages": 4,
        }
        if M <= E:
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
):
    from sglang.srt.layers.moe.fused_moe_triton import get_config

    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        configs = get_moe_configs(E, N, dtype)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, is_marlin)
    # TODO(HandH1998): Optimize the configs of block-wise quant.
    # NOTE(HandH1998): For block-wise quant,
    # BLOCK_K must be divisable by block_shape[1]
    # BLOCK_N and BLOCK_M has no requirements
    if block_shape is not None:
        config["BLOCK_SIZE_N"] = block_shape[0]
        config["BLOCK_SIZE_K"] = block_shape[1]
    return config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """Fused MoE computation using LMDeploy's implementation."""
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    M = hidden_states.size(0)
    E, N, _ = w1.shape
    num_experts = E

    intermediate_cache1 = torch.empty(
        (M, topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    def __get_sorted_idx(topk_ids: torch.Tensor):
        flatten_topk_ids = topk_ids.flatten()
        sorted_idx = flatten_topk_ids.argsort()

        exp_start, exp_end = get_start_end(flatten_topk_ids, sorted_idx, num_experts)
        return sorted_idx, exp_start, exp_end

    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    sorted_idx, exp_start, exp_end = __get_sorted_idx(topk_ids)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        get_config_dtype_str(
            hidden_states.dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
        ),
        M,
    )

    config = get_config_func(M)

    # gate and up

    fused_moe_kernel_launcher(
        hidden_states,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        weights=topk_weights,
        enable_weights=False,
        top_k=topk_ids.shape[1],
        num_tokens=M,
        expert_offset=0,
        reindex_a=True,
        reindex_c=False,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        config=config,
    )

    # activate
    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    # down
    fused_moe_kernel_launcher(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        weights=topk_weights,
        enable_weights=True,
        top_k=1,
        num_tokens=M,
        expert_offset=0,
        reindex_a=False,
        reindex_c=True,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        config=config,
    )

    ret = intermediate_cache3.sum(dim=1)
    return ret


def fused_moe_lmdeploy(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """fused moe."""
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=gating_output,
        use_grouped_topk=use_grouped_topk,
        top_k=topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
    )

    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )
