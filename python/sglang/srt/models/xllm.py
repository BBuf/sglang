# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only XLLM (IFM MoVA) model compatible with the XLLM HuggingFace
bridge checkpoints (``model_type: xllm``).

Architecture summary (mirrors ``xllm_bridges/vllm/xllm/modeling_xllm_vllm.py``):

- A dense prefix of ``num_dense_layers`` decoder layers followed by sparse
  layers. Sparse layers use MoVA attention (mixture-of-value-experts) when
  ``num_values > 0`` and a routed MoE MLP with optional shared experts.
- Grouped RMSNorm ("T5-style"): stored weights are offsets, the effective
  scale is ``1 + weight``; variance is computed per group.
- RoPE follows the native XLLM interleaved (GPT-J style) convention with an
  optional partial rotary dimension (``rope_head_dim < head_dim``). At weight
  load time the q/k projection rows (and q/k norm weights) are permuted from
  the HF half-split layout into the native interleaved layout so a standard
  ``is_neox_style=False`` rotary embedding reproduces native semantics.
- Optional attention output gate (``silu`` or ``softplus`` with beta=ln2).
- MoE / MoVA routing: scores are softmax/sigmoid of the router logits in
  float32; the (optional) router bias only affects expert *selection*, while
  the combined weights use the unbiased scores, optionally renormalized over
  top-k and scaled by ``router_scaling_factor``.
"""

import logging
import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    moe_expert_parallel_all_reduce,
    moe_tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


def _resolve_num_dense_layers(config: PretrainedConfig) -> int:
    """Port of xllm_bridges sparse_layout.resolve_num_dense_layers."""
    num_hidden_layers = config.num_hidden_layers
    num_experts = getattr(config, "num_experts", 0)
    if num_experts == 0:
        return num_hidden_layers

    num_dense_layers = getattr(config, "num_dense_layers", None)
    if num_dense_layers is not None:
        return max(0, min(int(num_dense_layers), num_hidden_layers))

    mlp_only_layers = getattr(config, "mlp_only_layers", None)
    if mlp_only_layers is None:
        return 0
    dense_prefix = 0
    for layer_idx in sorted(set(int(l) for l in mlp_only_layers)):
        if layer_idx != dense_prefix:
            break
        dense_prefix += 1
    return max(0, min(dense_prefix, num_hidden_layers))


def _is_sparse_layer(config: PretrainedConfig, layer_idx: int) -> bool:
    return (
        getattr(config, "num_experts", 0) > 0
        and layer_idx >= _resolve_num_dense_layers(config)
    )


def permute_qk_weight_to_interleaved(w: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Reorder per-head rows (or a 1-D per-head weight) from the HF half-split
    layout to the native XLLM interleaved layout:
    out[2i] = in[i], out[2i+1] = in[head_dim // 2 + i]."""
    if w.dim() == 1:
        return (
            w.reshape(-1, 2, head_dim // 2).transpose(1, 2).reshape(w.shape).contiguous()
        )
    cols = w.shape[1:]
    return (
        w.reshape(-1, 2, head_dim // 2, *cols).transpose(1, 2).reshape(w.shape).contiguous()
    )


def calc_router_weights(
    router_logits: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    score_func: str,
    top_k: int,
    scaling_factor: Optional[float],
    renormalize: Optional[bool] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact port of the XLLM router semantics (bias affects selection only)."""
    if score_func == "softmax":
        routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    elif score_func == "sigmoid":
        routing_scores = torch.sigmoid(router_logits.to(torch.float32))
    else:
        raise ValueError(f"Unsupported router score function: {score_func}")

    selection_scores = routing_scores
    if router_bias is not None:
        selection_scores = selection_scores + router_bias.to(selection_scores)

    selected_indices = torch.topk(selection_scores, top_k, dim=-1).indices
    routing_weights = torch.gather(routing_scores, dim=-1, index=selected_indices)
    if renormalize is None:
        renormalize = top_k > 1
    if renormalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    if scaling_factor is not None:
        routing_weights = routing_weights * scaling_factor
    if out_dtype is None:
        out_dtype = router_logits.dtype
    return routing_weights.to(out_dtype), selected_indices


class XllmRMSNorm(nn.Module):
    """Grouped RMSNorm with a +1 weight offset (T5 style), matching native
    XLLM GroupRMSNorm. Supports the fused (x, residual) calling convention."""

    def __init__(self, hidden_size: int, n_groups: int, eps: float = 1e-6):
        super().__init__()
        assert hidden_size % n_groups == 0
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Match the vLLM bridge exactly: the +1 offset is applied in the
        # activation dtype (bf16) before the float32 upcast.
        effective_weight = (self.weight.to(input_dtype) + 1.0).to(torch.float32)
        x = x.to(torch.float32)
        x = x.reshape(*x.shape[:-1], self.n_groups, -1)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.reshape(*x.shape[:-2], -1)
        x = effective_weight * x
        return x.to(input_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        if x.numel() == 0:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                return x, residual
            return x
        if residual is not None:
            x = x + residual.to(x.dtype)
            if post_residual_addition is not None:
                x = x + post_residual_addition.to(x.dtype)
            residual = x
            return self._norm(x), residual
        assert post_residual_addition is None
        return self._norm(x)


def _apply_attn_gate(
    attn_output: torch.Tensor, gate: torch.Tensor, gate_func: str
) -> torch.Tensor:
    if gate_func == "silu":
        gate = F.silu(gate)
    elif gate_func == "softplus":
        gate = F.softplus(gate, beta=math.log(2))
    else:
        raise ValueError(f"Unsupported attention gate function: {gate_func}")
    return attn_output * gate


class XllmAttention(nn.Module):
    """Dense GQA attention with grouped qk-norm, interleaved (partial) RoPE
    and an optional output gate."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.hidden_size = hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = (
            getattr(config, "head_dim", None) or hidden_size // self.total_num_heads
        )
        self.rope_head_dim = getattr(config, "rope_head_dim", None) or self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.apply_attn_gate = getattr(config, "apply_attn_gate", False)
        self.attn_gate_func = getattr(config, "attn_gate_func", "silu")
        self.query_key_norm = getattr(config, "query_key_norm", True)
        self.layer_id = layer_id

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.attn_gate_proj = (
            ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("attn_gate_proj", prefix),
            )
            if self.apply_attn_gate
            else None
        )

        # Weights are permuted into the native interleaved layout at load time,
        # so the GPT-J style (is_neox_style=False) rotary embedding matches
        # native XLLM RoPE, including the partial-rope case.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=getattr(config, "rope_theta", 10000.0),
            rope_scaling=None,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        if self.query_key_norm:
            self.q_norm = XllmRMSNorm(
                hidden_size=self.q_size,
                n_groups=self.num_heads,
                eps=config.rms_norm_eps,
            )
            self.k_norm = XllmRMSNorm(
                hidden_size=self.kv_size,
                n_groups=self.num_kv_heads,
                eps=config.rms_norm_eps,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.query_key_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q.contiguous(), k.contiguous())
        attn_output = self.attn(q, k, v, forward_batch)
        if self.attn_gate_proj is not None:
            gate, _ = self.attn_gate_proj(hidden_states)
            attn_output = _apply_attn_gate(attn_output, gate, self.attn_gate_func)
        output, _ = self.o_proj(attn_output)
        return output


class XllmMoVAAttention(nn.Module):
    """MoVA attention: the value projection is a routed mixture of value
    experts (per-expert ColumnParallelLinear + silu, combined with unbiased
    routing scores)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.hidden_size = hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % attn_tp_size != 0:
            raise ValueError(
                f"Attention heads {self.total_num_heads} must be divisible by "
                f"TP size {attn_tp_size}."
            )
        if self.total_num_kv_heads % attn_tp_size != 0:
            raise ValueError(
                "MoVA support requires tensor parallel size not to exceed the "
                "number of KV heads."
            )
        self.num_heads = self.total_num_heads // attn_tp_size
        self.num_kv_heads = self.total_num_kv_heads // attn_tp_size
        self.head_dim = (
            getattr(config, "head_dim", None) or hidden_size // self.total_num_heads
        )
        self.rope_head_dim = getattr(config, "rope_head_dim", None) or self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.num_values = config.num_values
        self.top_k = config.num_values_per_tok
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor
        self.apply_attn_gate = getattr(config, "apply_attn_gate", False)
        self.attn_gate_func = getattr(config, "attn_gate_func", "silu")
        self.query_key_norm = getattr(config, "query_key_norm", True)
        self.layer_id = layer_id
        qkv_bias = getattr(config, "attention_bias", False)

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_router = ReplicatedLinear(
            hidden_size,
            self.num_values,
            bias=getattr(config, "moe_gate_bias", False),
            skip_bias_add=True,
            quant_config=quant_config,
            prefix=add_prefix("v_router", prefix),
        )
        self.v_experts = nn.ModuleList(
            [
                ColumnParallelLinear(
                    hidden_size,
                    self.total_num_kv_heads * self.head_dim,
                    bias=False,
                    quant_config=quant_config,
                    tp_rank=attn_tp_rank,
                    tp_size=attn_tp_size,
                    prefix=add_prefix(f"v_experts.{value}", prefix),
                )
                for value in range(self.num_values)
            ]
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.attn_gate_proj = (
            ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("attn_gate_proj", prefix),
            )
            if self.apply_attn_gate
            else None
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=getattr(config, "rope_theta", 10000.0),
            rope_scaling=None,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        if self.query_key_norm:
            self.q_norm = XllmRMSNorm(
                hidden_size=self.q_size,
                n_groups=self.num_heads,
                eps=config.rms_norm_eps,
            )
            self.k_norm = XllmRMSNorm(
                hidden_size=self.kv_size,
                n_groups=self.num_kv_heads,
                eps=config.rms_norm_eps,
            )

        # Whether torch._grouped_mm is usable is probed lazily on the first
        # (eager warmup) forward, before any CUDA graph capture.
        self._use_grouped_mm: Optional[bool] = None

    def _combine_value_experts_grouped(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        top_k = selected_indices.shape[-1]
        flat_indices = selected_indices.reshape(-1)
        sorted_indices = torch.argsort(flat_indices, stable=True)
        permute_indices = sorted_indices // top_k
        expert_weight = torch.stack([e.weight for e in self.v_experts], dim=0)
        permuted = hidden_states.index_select(0, permute_indices)
        classes = torch.arange(
            self.num_values, device=flat_indices.device, dtype=flat_indices.dtype
        )
        group_sizes = (
            (flat_indices.unsqueeze(-1) == classes).to(torch.int32).sum(dim=0)
        )
        offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
        expert_states = F.silu(
            torch._grouped_mm(permuted, expert_weight.transpose(1, 2), offs=offsets)
        )
        inverse_indices = torch.argsort(sorted_indices, stable=True)
        expert_states = expert_states.index_select(0, inverse_indices)
        expert_states = expert_states.view(num_tokens, top_k, -1)
        return (
            expert_states * routing_weights.to(expert_states.dtype).unsqueeze(2)
        ).sum(dim=1)

    def _combine_value_experts_sequential(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        top_k = selected_indices.shape[-1]
        expert_states = hidden_states.new_zeros(num_tokens, top_k, self.kv_size)
        for expert_idx in range(self.num_values):
            token_positions, topk_positions = torch.where(
                selected_indices == expert_idx
            )
            if token_positions.numel() == 0:
                continue
            out, _ = self.v_experts[expert_idx](hidden_states[token_positions])
            expert_states[token_positions, topk_positions] = F.silu(out).to(
                expert_states.dtype
            )
        return (
            expert_states * routing_weights.to(expert_states.dtype).unsqueeze(2)
        ).sum(dim=1)

    def _compute_value_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = F.linear(hidden_states, self.v_router.weight)
        router_bias = getattr(self.v_router, "bias", None)
        routing_weights, selected_values = calc_router_weights(
            router_logits=router_logits,
            router_bias=router_bias,
            score_func=self.router_score_func,
            top_k=self.top_k,
            scaling_factor=self.router_scaling_factor,
        )
        if self._use_grouped_mm is None:
            try:
                value_states = self._combine_value_experts_grouped(
                    hidden_states, routing_weights, selected_values
                )
                self._use_grouped_mm = True
                return value_states
            except (RuntimeError, AttributeError) as e:
                logger.warning(
                    "torch._grouped_mm unavailable for MoVA value experts (%s); "
                    "falling back to the sequential path. CUDA graph capture "
                    "requires the grouped path; use --disable-cuda-graph.",
                    e,
                )
                self._use_grouped_mm = False
        if self._use_grouped_mm:
            return self._combine_value_experts_grouped(
                hidden_states, routing_weights, selected_values
            )
        return self._combine_value_experts_sequential(
            hidden_states, routing_weights, selected_values
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        value_states = self._compute_value_states(hidden_states)

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        if self.query_key_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q.contiguous(), k.contiguous())
        attn_output = self.attn(q, k, value_states.contiguous(), forward_batch)
        if self.attn_gate_proj is not None:
            gate, _ = self.attn_gate_proj(hidden_states)
            attn_output = _apply_attn_gate(attn_output, gate, self.attn_gate_func)
        output, _ = self.o_proj(attn_output)
        return output


class XllmSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_parallel().moe_tp_size
        self.ep_size = get_parallel().moe_ep_size
        self.layer_id = layer_id
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.top_k = config.num_experts_per_tok
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=getattr(config, "moe_gate_bias", False),
            skip_bias_add=True,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.Renormalize,
        )

        self.num_shared_experts = getattr(config, "num_shared_experts", 0)
        if self.num_shared_experts > 0:
            self.shared_experts = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * self.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, router_bias = self.gate(hidden_states)
        routing_weights, selected_experts = calc_router_weights(
            router_logits=router_logits,
            router_bias=router_bias,
            score_func=self.router_score_func,
            top_k=self.top_k,
            scaling_factor=self.router_scaling_factor,
            renormalize=self.norm_topk_prob,
        )
        topk_output = StandardTopKOutput(
            topk_weights=routing_weights.to(torch.float32),
            topk_ids=selected_experts.to(torch.int32),
            router_logits=router_logits,
        )
        # NOTE: shared experts must run before the routed experts because the
        # fused MoE kernel may write its output into `hidden_states` in place.
        shared_output = None
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)
        final_hidden_states = self.experts(hidden_states, topk_output)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False
        ):
            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True
        ):
            final_hidden_states = moe_tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)


class XllmDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        is_sparse = _is_sparse_layer(config, layer_id)
        is_mova = is_sparse and getattr(config, "num_values", 0) > 0
        if is_mova:
            self.self_attn = XllmMoVAAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.self_attn = XllmAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )

        self.is_layer_sparse = is_sparse
        if is_sparse:
            self.mlp = XllmSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        num_dense = _resolve_num_dense_layers(config)

        def layer_is_sparse(idx: int) -> bool:
            return 0 <= idx < config.num_hidden_layers and _is_sparse_layer(
                config, idx
            )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_sparse,
            is_previous_layer_sparse=layer_is_sparse(layer_id - 1),
            is_next_layer_sparse=layer_is_sparse(layer_id + 1),
        )

        layernorm_num_groups = getattr(config, "layernorm_num_groups", 1)
        self.input_layernorm = XllmRMSNorm(
            config.hidden_size, layernorm_num_groups, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = XllmRMSNorm(
            config.hidden_size, layernorm_num_groups, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        if self.is_layer_sparse:
            hidden_states = self.mlp(hidden_states, forward_batch)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class XllmModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: XllmDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = XllmRMSNorm(
                config.hidden_size,
                getattr(config, "layernorm_num_groups", 1),
                eps=config.rms_norm_eps,
            )
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class XllmForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = XllmModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def _maybe_permute_qk_weight(
        self, name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        """Permute q/k projection rows and q/k norm weights from the HF
        half-split layout to the native interleaved layout (see module
        docstring). v/o/gate weights are layout-independent."""
        if not name.endswith(
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            )
        ):
            return loaded_weight
        head_dim = (
            getattr(self.config, "head_dim", None)
            or self.config.hidden_size // self.config.num_attention_heads
        )
        return permute_qk_weight_to_interleaved(loaded_weight, head_dim)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            loaded_weight = self._maybe_permute_qk_weight(name, loaded_weight)

            # Direct parameter hits first: MoVA modules keep separate
            # q_proj/k_proj/v_router/v_experts parameters whose checkpoint
            # names would otherwise be captured by the stacked mapping.
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name not in params_dict:
                        continue
                    param = params_dict[mapped_name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        logger.warning("Parameter %s not found in params_dict", name)
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = XllmForCausalLM
