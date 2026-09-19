Cumulative diagnostic times, not E2E. The automatic FP8 replacement suggestion also groups an unquantized BF16 GEMM and is not used. Existing QK-norm/RoPE fusion is already active.

Triage View
Mode: single-trace
Framework: SGLang
Input traces: /campaign/artifacts/fasth3/fp8-swiglu-profile/traces/denoise-step2.trace.json.gz

Kernel Table
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| _attn_fwd_sparse | other | 1276.02 ms | 39.3% | 50 | python/sglang/multimodal_gen/runtime/layers/attention/backends/vsa_h3_kernels.py:101 vsa_h3_block_sparse_attn_forward | cuLaunchKernelEx |
| nvjet_sm90_qqtst_128x128_128x6_1x2_h_bz_ovscale_TNT | gemm | 951.03 ms | 29.3% | 200 | python/sglang/kernels/ops/gemm/__init__.py:125 forward_native | aten::_scaled_mm |
| nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT | gemm | 197.67 ms | 6.1% | 50 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::mm |
| ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>) | communication | 153.86 ms | 4.7% | 50 | python/sglang/multimodal_gen/runtime/layers/usp.py:97 _usp_all_to_all_single | record_param_comms |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | memory | 89.69 ms | 2.8% | 200 | python/sglang/multimodal_gen/runtime/layers/usp.py:141 _ipc_varlen_fast | aten::copy_ |
| void sglang::per_token_quant_fp8_warp_kernel<__nv_bfloat16, 8> | quantize | 70.37 ms | 2.2% | 200 | python/sglang/kernels/ops/quantization/__init__.py:76 sgl_per_token_quant_fp8 | sglang::per_token_quant_fp8 |
| void sglang::fused_qknorm_rope_warp<128l, 96l, true, true, __nv_bfloat16, __nv_bfloat16, true, false, false, long, false> | rope | 55.52 ms | 1.7% | 50 | python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1045 forward | sglang::fused_inplace_qknorm_rope |
| _pack_tiles_kernel | other | 54.11 ms | 1.7% | 50 | python/sglang/multimodal_gen/runtime/layers/attention/backends/vsa_h3_kernels.py:233 vsa_h3_pack_tiles | cuLaunchKernelEx |
| void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, true, false, true, false> | activation | 45.13 ms | 1.4% | 50 | python/sglang/kernels/ops/activation/activation.py:180 silu_and_mul_with_activation_rounding | sglang::_run_activation_with_rounding_inplace |
| _pack_qkv_destination_major_kernel | other | 37.29 ms | 1.2% | 50 | python/sglang/kernels/ops/diffusion/layout/ulysses_qkv_triton.py:55 pack_qkv_destination_major | cuLaunchKernelEx |
| Memcpy PtoP (Device -> Device) | memory | 33.66 ms | 1.0% | 50 | python/sglang/multimodal_gen/runtime/distributed/device_communicators/ipc_a2a.py:181 exchange | aten::copy_ |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Confirmed | 1153.32 ms | 35.6% | nvjet_sm90_qqtst_128x128_128x6_1x2_h_bz_ovscale_TNT (29.3%)<br>nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT (6.1%) | forward_aot @ python/sglang/kernels/ops/gemm/__init__.py:149<br>forward_native @ python/sglang/kernels/ops/gemm/__init__.py:125<br>apply_unquantized_linear @ python/sglang/multimodal_gen/runtime/layers/linear.py:134<br>_topk_tile_lists @ python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn_h3.py:228 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | Matches an open upstream path (35.6% related GPU time). Open SGLang PR replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RMSNorm + RoPE | Confirmed | 55.52 ms | 1.7% | void sglang::fused_qknorm_rope_warp<128l, 96l, true, true, __nv_bfloat16, __nv_bfloat16, true, false, false, long, false> (1.7%) | forward @ python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1045 | python/sglang/jit_kernel/fused_qknorm_rope.py<br>python/sglang/srt/models/qwen3_moe.py | `Fused QK RMSNorm + RoPE` is present in this trace (1.7% related GPU time). SGLang has a fused QK-norm plus RoPE kernel family. |
| Fused activation-and-mul (SwiGLU / GeGLU) | Confirmed | 45.13 ms | 1.4% | void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, true, false, true, false> (1.4%) | silu_and_mul_with_activation_rounding @ python/sglang/kernels/ops/activation/activation.py:180 | python/sglang/srt/layers/activation.py | `Fused activation-and-mul (SwiGLU / GeGLU)` is present in this trace (1.4% related GPU time). Packed MLP activation and multiply already has dedicated fused ops. |
