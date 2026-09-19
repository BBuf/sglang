Diagnostic cumulative GPU times. The automated nvjet-to-FP8 suggestion is rejected: these GEMMs are BF16.

Triage View
Mode: single-trace
Framework: SGLang
Input traces: /Users/bbuf/工作目录/Common/diffusion-h200-eight-pr-20260920/artifacts/longcat-image-edit-turbo/combined-bcg-profile/traces/denoise-step2.trace.json.gz

Kernel Table
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<176>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, false, false, false, false, false, true, true, false, false, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<176> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, false, false, false, false>, flash::StaticPersistentTileScheduler<false> > > > | gemm | 48.98 ms | 31.9% | 30 | python/sglang/kernels/ops/attention/flash_attention_v3.py:19 _call_fa3_kernel | sgl_kernel::fwd |
| nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_bias_TNT | gemm | 45.81 ms | 29.9% | 80 | python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 replay | cudaGraphLaunch |
| nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN | gemm | 38.78 ms | 25.3% | 80 | python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 replay | cudaGraphLaunch |
| void sglang::fused_qknorm_rope_warp<128l, 128l, false, true, __nv_bfloat16, float, true, false, true, long, false> | rope | 3.51 ms | 2.3% | 40 | python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 replay | cudaGraphLaunch |
| void sglang::gelu_tanh_cat_kernel<__nv_bfloat16, 8> | activation | 3.26 ms | 2.1% | 20 | python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 replay | cudaGraphLaunch |
| _layernorm_modulate_kernel | norm | 2.39 ms | 1.6% | 60 | python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 replay | cudaGraphLaunch |
| nvjet_sm90_tst_192x112_64x5_1x2_h_bz_coopB_bias_TNT | gemm | 2.24 ms | 1.5% | 51 | python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 replay | cudaGraphLaunch |
| void sglang::residual_gate_add::(anonymous namespace)::residual_gate_add_broadcast_kernel<__nv_bfloat16, 8> | communication | 1.65 ms | 1.1% | 60 | python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 replay | cudaGraphLaunch |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Confirmed | 89.21 ms | 58.2% | nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_bias_TNT (29.9%)<br>nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN (25.3%)<br>nvjet_sm90_tst_192x112_64x5_1x2_h_bz_coopB_bias_TNT (1.5%) | replay @ python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | Matches an open upstream path (58.2% related GPU time). Open SGLang PR replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused MoE router / top-k / softcapping | Confirmed | 4.91 ms | 3.2% | void sglang::residual_gate_add::(anonymous namespace)::residual_gate_add_broadcast_kernel<__nv_bfloat16, 8> (1.1%)<br>void sglang::gelu_tanh_cat_kernel<__nv_bfloat16, 8> (2.1%) | replay @ python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 | python/sglang/srt/layers/moe/router.py | Split kernels in this family take 3.2% of GPU time. This tree already has a matching path. MoE routing already has fused router, softcap, and top-k kernels. |
| Fused QK RMSNorm + RoPE | Confirmed | 3.51 ms | 2.3% | void sglang::fused_qknorm_rope_warp<128l, 128l, false, true, __nv_bfloat16, float, true, false, true, long, false> (2.3%) | replay @ python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:284 | python/sglang/jit_kernel/fused_qknorm_rope.py<br>python/sglang/srt/models/qwen3_moe.py | `Fused QK RMSNorm + RoPE` is present in this trace (2.3% related GPU time). SGLang has a fused QK-norm plus RoPE kernel family. |
