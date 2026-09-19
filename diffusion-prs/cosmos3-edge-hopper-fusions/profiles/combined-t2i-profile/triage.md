Diagnostic cumulative GPU times. The automated nvjet-to-FP8 suggestion is rejected: these GEMMs are BF16.

Triage View
Mode: single-trace
Framework: SGLang
Input traces: /campaign/artifacts/cosmos3-edge/combined-t2i-profile/traces/denoise-step2.trace.json.gz

Kernel Table
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_sm90_tst_64x208_64x6_2x1_v_bz_splitK_TNT | gemm | 1.50 ms | 24.5% | 56 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::mm |
| nvjet_sm90_tst_384x80_64x3_4x1_v_bz_coopA_TNT | gemm | 1.47 ms | 24.0% | 56 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::mm |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<176>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, false, false, false, false, false, true, true, true, false, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<176> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, false, true, false, false>, flash::StaticPersistentTileScheduler<false> > > > | gemm | 0.70 ms | 11.4% | 56 | python/sglang/kernels/ops/attention/flash_attention_v3.py:19 _call_fa3_kernel | sgl_kernel::fwd |
| nvjet_sm90_tst_128x104_64x7_1x2_h_bz_TNT | gemm | 0.68 ms | 11.1% | 56 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::mm |
| nvjet_sm90_tst_64x104_64x10_2x1_v_bz_TNT | gemm | 0.46 ms | 7.6% | 56 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::mm |
| kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64204820481_tensorptrbf16gmemalign128oi64204820481_tensorptrbf16gme_0 | gemm | 0.32 ms | 5.3% | 110 | python/sglang/multimodal_gen/runtime/layers/layernorm.py:123 forward_cuda | cudaLaunchKernelExC |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.27 ms | 4.3% | 56 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::mm |
| void sglang::fused_qknorm_rope_warp<128l, 128l, true, true, __nv_bfloat16, __nv_bfloat16, true, true, false, long, false> | rope | 0.25 ms | 4.1% | 56 | python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py:282 _apply_qwen3_qk_norm_rope_pack_kv | sglang::fused_qknorm_rope_pack_kv |
| void sglang::act_kernel<__nv_bfloat16, (sglang::ActivationKind)3, true> | other | 0.21 ms | 3.5% | 56 | python/sglang/kernels/ops/activation/activation.py:144 run_unary_activation | sglang::_run_unary_activation_inplace |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Fused QK RMSNorm + RoPE | Confirmed | 0.25 ms | 4.1% | void sglang::fused_qknorm_rope_warp<128l, 128l, true, true, __nv_bfloat16, __nv_bfloat16, true, true, false, long, false> (4.1%) | _apply_qwen3_qk_norm_rope_pack_kv @ python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py:282 | python/sglang/jit_kernel/fused_qknorm_rope.py<br>python/sglang/srt/models/qwen3_moe.py | `Fused QK RMSNorm + RoPE` is present in this trace (4.1% related GPU time). SGLang has a fused QK-norm plus RoPE kernel family. |
| NSA fused quantize + indexed K-cache store | Confirmed | 4.45 ms | 72.6% | nvjet_sm90_tst_64x208_64x6_2x1_v_bz_splitK_TNT (24.5%)<br>nvjet_sm90_tst_384x80_64x3_4x1_v_bz_coopA_TNT (24.0%)<br>nvjet_sm90_tst_128x104_64x7_1x2_h_bz_TNT (11.1%) | apply_unquantized_linear @ python/sglang/multimodal_gen/runtime/layers/linear.py:134<br>multistep_uni_c_bh_update @ python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_unipc_multistep.py:490<br>multistep_uni_p_bh_update @ python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_unipc_multistep.py:351<br>_sigma_to_alpha_sigma_t @ python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_unipc_multistep.py:277 | python/sglang/jit_kernel/fused_store_index_cache.py<br>python/sglang/srt/layers/attention/nsa/nsa_indexer.py | Split kernels in this family take 72.6% of GPU time. This tree already has a matching path. NSA already has a fused quantize-and-indexed-store kernel family. |
| Fused residual add + RMSNorm | Confirmed | 0.32 ms | 5.3% | kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64204820481_tensorptrbf16gmemalign128oi64204820481_tensorptrbf16gme_0 (5.3%) | forward_cuda @ python/sglang/multimodal_gen/runtime/layers/layernorm.py:123 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | `Fused residual add + RMSNorm` is present in this trace (5.3% related GPU time). Residual add plus RMSNorm already has fused implementations across several backends. |
