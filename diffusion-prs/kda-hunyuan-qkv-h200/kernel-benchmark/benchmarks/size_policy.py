"""Per-kernel workload size classification.

A workload is either ``small`` (launch/occupancy-overhead bound, useful for
regression detection) or ``large`` (compute/memory-bandwidth bound, the
production-representative regime). The split axis and threshold are chosen
per kernel family because "small" for a decode kernel is not comparable to
"small" for a prefill or MoE kernel.

The same rules drive three consumers, so they never disagree:
  * the dataset tagger that writes ``size_class`` onto every workload row,
  * the dataset validator that fail-closes on missing/wrong tags, and
  * the benchmark summary that reports all / large / small geomeans.

Classification is deterministic from the workload axes. Anything the rules
cannot classify raises ``UnknownWorkload`` so callers fail closed rather than
silently dropping a row into the wrong bucket.
"""

from __future__ import annotations

SMALL = "small"
LARGE = "large"
VALID_SIZE_CLASSES = (SMALL, LARGE)

# Runner-supplied names carry provenance suffixes; strip them to the definition.
_NAME_SUFFIXES = ("_official", "_large_embedded", "_large_live")


class UnknownWorkload(ValueError):
    """Raised when a workload cannot be classified by the size rules."""


def normalize_name(name: str) -> str:
    """Reduce a runner name (e.g. ``moe_..._large_live``) to its definition."""
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _require(axes: dict, key: str, definition: str):
    if key not in axes or not isinstance(axes[key], (int, float)):
        raise UnknownWorkload(
            f"{definition}: missing or non-numeric size axis '{key}' in axes={axes}"
        )
    return axes[key]


# Each rule takes the workload axes and returns True when the workload is large.
# Thresholds are inclusive on the large side and are anchored to the observed
# official distribution plus the authors' embedded large shapes.
_RULES = {
    # prefill: total tokens drive compute; embedded large 16384/32768
    "gdn_prefill_qk4_v8_d128_k_last": lambda a, d: (
        _require(a, "total_seq_len", d) >= 4096
    ),
    # KDA forward/backward: equal-length batches; sequence length drives the
    # chunked-scan depth, so the 4096/8192 windows are the production regime
    "kda_forward_k128_v128_bf16": lambda a, d: _require(a, "seq_len", d) >= 4096,
    "kda_backward_k128_v128_bf16": lambda a, d: _require(a, "seq_len", d) >= 4096,
    "gdn2_forward_h16_k128_v128_bf16": lambda a, d: (
        _require(a, "total_tokens", d) >= 4096
    ),
    "gated_oja_forward_k128_v128_bf16": lambda a, d: (
        _require(a, "total_tokens", d) >= 4096
    ),
    "iplr_forward_k128_v128_bf16": lambda a, d: _require(a, "total_tokens", d) >= 4096,
    "comba_forward_h6_k256_v512_bf16": lambda a, d: (
        _require(a, "total_tokens", d) >= 4096
    ),
    "gsa_forward_h4_k512_v512_m64_bf16": lambda a, d: (
        _require(a, "total_tokens", d) >= 4096
    ),
    "gated_delta_product_forward_h6_k256_v512_n2_bf16": lambda a, d: (
        _require(a, "total_tokens", d) >= 4096
    ),
    "mesa_net_forward_h16_k128_v128_cg30_bf16": lambda a, d: (
        _require(a, "total_tokens", d) >= 4096
    ),
    "log_linear_attention_forward_h64_k128_v64_l15_bf16": lambda a, d: (
        _require(a, "total_tokens", d) >= 4096
    ),
    # sparse MLA: query tokens drive compute, pages drive KV footprint
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64": lambda a, d: (
        _require(a, "num_tokens", d) >= 16 or _require(a, "num_pages", d) >= 16384
    ),
    # MoE: token count drives grouped-GEMM work; official mostly <=80, large 32768
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": (
        lambda a, d: _require(a, "seq_len", d) >= 512
    ),
    # Qwen3.8-27B NVFP4 (sm_120), captured under DSpark. The FP4 GEMM task is an
    # exhaustive M=1..16 sweep split evenly at the two production anchors:
    # M=1 ordinary decode is in the small half and M=9 DSpark verify starts the
    # large half. The other two Qwen tasks retain their captured tier rules.
    # The captures imported from KDA-Pilot. One rule for all of them: these captures come
    # from serving, where a row is either a decode step of a handful of tokens or a
    # prefill/verify window of many, and 8 tokens is where the kernels stop being
    # launch-bound. `tokens` is the adapter's own axis (see _pilot_adapter.axes).
    "glm45__fp8_fused_moe": lambda a, d: _require(a, "tokens", d) >= 8,
    "kimi_k3__tgv_bf16_tiny_gemm": lambda a, d: _require(a, "tokens", d) >= 8,
    "lfm25__triton_fused_moe": lambda a, d: _require(a, "tokens", d) >= 8,
    "qwen38_nvfp4__fp4_w4a4_skinny_gemm": lambda a, d: _require(a, "tokens", d) >= 9,
    "qwen38_nvfp4__fp8_verify_skinny_gemm": lambda a, d: _require(a, "tokens", d) >= 9,
    # Every verify forward is exactly T=9, so this family has one tier only.
    "qwen38_nvfp4__gdn_sigmoid_gating_verify": lambda a, d: (
        _require(a, "tokens", d) >= 9
    ),
    "glm47_mla_decode_grouped_h20_ckv512_kpe64": lambda a, d: (
        _require(a, "len_kv_indices", d) >= 8192
    ),
    "qwen3next_gdn_packed_decode_hv4_d128": lambda a, d: (
        _require(a, "num_seqs", d) >= 8
    ),
    # Shallow: the task exists to speed up decode, so the decode tier (M <= 64) is
    # the primary "large" bucket and prefill (M >= 256) is the guardrail. Inverted
    # on purpose; the adapter's size_class carries the same rule.
    "shallow__fp8_block32_dense_gemm": lambda a, d: _require(a, "tokens", d) <= 64,
}

# Minimum number of small workloads that must remain per definition so that
# launch-overhead regressions stay observable after pruning.
MIN_SMALL_COVERAGE = {
    "gdn_prefill_qk4_v8_d128_k_last": 8,
    "kda_forward_k128_v128_bf16": 4,
    "kda_backward_k128_v128_bf16": 4,
    "gsa_forward_h4_k512_v512_m64_bf16": 1,
    "gated_delta_product_forward_h6_k256_v512_n2_bf16": 1,
    "mesa_net_forward_h16_k128_v128_cg30_bf16": 1,
    "log_linear_attention_forward_h64_k128_v64_l15_bf16": 1,
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64": 4,
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": 6,
    # The imported packages: the floor is what the capture actually shipped, so pruning
    # cannot quietly drop a tier.
    "glm45__fp8_fused_moe": 0,
    "kimi_k3__tgv_bf16_tiny_gemm": 10,
    "lfm25__triton_fused_moe": 0,
    "qwen38_nvfp4__fp4_w4a4_skinny_gemm": 24,
    "qwen38_nvfp4__fp8_verify_skinny_gemm": 4,
    # DSPARK block 8 makes T=9 exact for every verify forward: this family has no
    # second tier, so its small geomean is empty by construction. A zero here is
    # the honest floor, not a missing capture.
    "qwen38_nvfp4__gdn_sigmoid_gating_verify": 0,
    "glm47_mla_decode_grouped_h20_ckv512_kpe64": 5,
    "qwen3next_gdn_packed_decode_hv4_d128": 5,
    "shallow__fp8_block32_dense_gemm": 12,
}

# Minimum number of large workloads that must be checked in per ranked family.
# The large geomean is the primary ranking metric, so a ranked family must never
# have zero checked-in large workloads (which would make its ranking meaningless).
MIN_LARGE_COVERAGE = {
    "gdn_prefill_qk4_v8_d128_k_last": 2,
    "kda_forward_k128_v128_bf16": 2,
    "kda_backward_k128_v128_bf16": 2,
    "gdn2_forward_h16_k128_v128_bf16": 2,
    "gated_oja_forward_k128_v128_bf16": 2,
    "iplr_forward_k128_v128_bf16": 2,
    "comba_forward_h6_k256_v512_bf16": 2,
    "gsa_forward_h4_k512_v512_m64_bf16": 2,
    "gated_delta_product_forward_h6_k256_v512_n2_bf16": 2,
    "mesa_net_forward_h16_k128_v128_cg30_bf16": 2,
    "log_linear_attention_forward_h64_k128_v64_l15_bf16": 2,
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64": 4,
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": 1,
    "glm45__fp8_fused_moe": 17,
    "kimi_k3__tgv_bf16_tiny_gemm": 12,
    "lfm25__triton_fused_moe": 14,
    "qwen38_nvfp4__fp4_w4a4_skinny_gemm": 24,
    "qwen38_nvfp4__fp8_verify_skinny_gemm": 4,
    "qwen38_nvfp4__gdn_sigmoid_gating_verify": 3,
    "glm47_mla_decode_grouped_h20_ckv512_kpe64": 2,
    "qwen3next_gdn_packed_decode_hv4_d128": 3,
    "shallow__fp8_block32_dense_gemm": 18,
}


def known_definitions() -> tuple:
    return tuple(_RULES)


def classify(definition: str, axes: dict) -> str:
    """Return ``"small"`` or ``"large"`` for a workload, or raise UnknownWorkload."""
    rule = _RULES.get(normalize_name(definition))
    if rule is None:
        raise UnknownWorkload(f"no size rule for definition '{definition}'")
    return LARGE if rule(axes, definition) else SMALL


# H200 diffusion campaign rules.
_RULES["diffusion_h200__wan_causal_cache"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__wan_causal_cache"] = 4
MIN_LARGE_COVERAGE["diffusion_h200__wan_causal_cache"] = 4
_RULES["diffusion_h200__wan_nearest_upsample"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__wan_nearest_upsample"] = 3
MIN_LARGE_COVERAGE["diffusion_h200__wan_nearest_upsample"] = 6
_RULES["diffusion_h200__wan_temb_slices"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__wan_temb_slices"] = 2
MIN_LARGE_COVERAGE["diffusion_h200__wan_temb_slices"] = 6
_RULES["diffusion_h200__ltx2_ada_values"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__ltx2_ada_values"] = 4
MIN_LARGE_COVERAGE["diffusion_h200__ltx2_ada_values"] = 4
_RULES["diffusion_h200__zimage_native_rmsnorm"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__zimage_native_rmsnorm"] = 4
MIN_LARGE_COVERAGE["diffusion_h200__zimage_native_rmsnorm"] = 6
_RULES["diffusion_h200__h3_indexed_modulation"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__h3_indexed_modulation"] = 6
MIN_LARGE_COVERAGE["diffusion_h200__h3_indexed_modulation"] = 6
_RULES["diffusion_h200__sana_conv_post"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__sana_conv_post"] = 6
MIN_LARGE_COVERAGE["diffusion_h200__sana_conv_post"] = 12
_RULES["diffusion_h200__joint_qkv_cat"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__joint_qkv_cat"] = 2
MIN_LARGE_COVERAGE["diffusion_h200__joint_qkv_cat"] = 4
_RULES["diffusion_h200__varlen_pack"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__varlen_pack"] = 6
MIN_LARGE_COVERAGE["diffusion_h200__varlen_pack"] = 3
_RULES["diffusion_h200__swiglu_bitexact"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__swiglu_bitexact"] = 4
MIN_LARGE_COVERAGE["diffusion_h200__swiglu_bitexact"] = 6
_RULES["diffusion_h200__complex_rope"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__complex_rope"] = 3
MIN_LARGE_COVERAGE["diffusion_h200__complex_rope"] = 5
_RULES["diffusion_h200__rotate_half_rope"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__rotate_half_rope"] = 3
MIN_LARGE_COVERAGE["diffusion_h200__rotate_half_rope"] = 5
_RULES["diffusion_h200__hunyuan_qkv_rope_pack"] = lambda a, d: (
    _require(a, "elements", d) >= 1048576
)
MIN_SMALL_COVERAGE["diffusion_h200__hunyuan_qkv_rope_pack"] = 2
MIN_LARGE_COVERAGE["diffusion_h200__hunyuan_qkv_rope_pack"] = 4
