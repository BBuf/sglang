Cumulative diagnostic times, not E2E. The automatic FP8 replacement suggestion also groups an unquantized BF16 GEMM and is not used. Existing QK-norm/RoPE fusion is already active.

{
  "source": "/campaign/artifacts/fasth3/fp8-scoped-baseline-profile/traces/cba3c5ee-9081-41e1-9ebf-de49eb64c1a8-2_steps-global-rank0.trace.json.gz",
  "slice": "/campaign/artifacts/fasth3/fp8-scoped-baseline-profile/traces/denoise-step2.trace.json.gz",
  "source_event_count": 325853,
  "slice_event_count": 107759,
  "start_us": 6879423274948.656,
  "end_us": 6879426606362.728,
  "window_ms": 3331.4140712890626,
  "crossing_kernel_count": 0,
  "boundary_rule": "Second recorded MiniMaxH3DiTModel start to third; stage synchronization enabled",
  "swiglu_calls": 50,
  "swiglu_kernel_count": 100,
  "swiglu_gpu_ms": 157.844783,
  "swiglu_kernels": [
    {
      "name": "void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})",
      "count": 50,
      "total_ms": 81.01198299999999
    },
    {
      "name": "void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})",
      "count": 50,
      "total_ms": 76.83280000000003
    }
  ]
}
Triage View
Mode: single-trace
Framework: SGLang
Input traces: /campaign/artifacts/fasth3/fp8-scoped-baseline-profile/traces/denoise-step2.trace.json.gz

Kernel Table
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| _attn_fwd_sparse | other | 1255.63 ms | 37.8% | 50 | python/sglang/multimodal_gen/runtime/layers/attention/backends/vsa_h3_kernels.py:101 vsa_h3_block_sparse_attn_forward | cuLaunchKernelEx |
| nvjet_sm90_qqtst_128x128_128x6_1x2_h_bz_ovscale_TNT | gemm | 953.60 ms | 28.7% | 200 | python/sglang/kernels/ops/gemm/__init__.py:125 forward_native | aten::_scaled_mm |
| nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT | gemm | 199.19 ms | 6.0% | 50 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::mm |
| ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>) | communication | 143.62 ms | 4.3% | 50 | python/sglang/multimodal_gen/runtime/layers/usp.py:97 _usp_all_to_all_single | record_param_comms |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | memory | 89.37 ms | 2.7% | 200 | python/sglang/multimodal_gen/runtime/layers/usp.py:141 _ipc_varlen_fast | aten::copy_ |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | activation | 81.01 ms | 2.4% | 50 | python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:428 _silu_mul | aten::silu |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}> | elementwise | 76.83 ms | 2.3% | 50 | python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:428 _silu_mul | aten::mul |
| void sglang::per_token_quant_fp8_warp_kernel<__nv_bfloat16, 8> | quantize | 70.36 ms | 2.1% | 200 | python/sglang/kernels/ops/quantization/__init__.py:76 sgl_per_token_quant_fp8 | sglang::per_token_quant_fp8 |
| void sglang::fused_qknorm_rope_warp<128l, 96l, true, true, __nv_bfloat16, __nv_bfloat16, true, false, false, long, false> | rope | 56.71 ms | 1.7% | 50 | python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1041 forward | sglang::fused_inplace_qknorm_rope |
| _pack_tiles_kernel | other | 53.72 ms | 1.6% | 50 | python/sglang/multimodal_gen/runtime/layers/attention/backends/vsa_h3_kernels.py:233 vsa_h3_pack_tiles | cuLaunchKernelEx |
| _pack_qkv_destination_major_kernel | other | 37.52 ms | 1.1% | 50 | python/sglang/kernels/ops/diffusion/layout/ulysses_qkv_triton.py:55 pack_qkv_destination_major | cuLaunchKernelEx |
| Memcpy PtoP (Device -> Device) | memory | 33.66 ms | 1.0% | 50 | python/sglang/multimodal_gen/runtime/distributed/device_communicators/ipc_a2a.py:181 exchange | aten::copy_ |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Confirmed | 1157.40 ms | 34.8% | nvjet_sm90_qqtst_128x128_128x6_1x2_h_bz_ovscale_TNT (28.7%)<br>nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT (6.0%) | forward_aot @ python/sglang/kernels/ops/gemm/__init__.py:149<br>forward_native @ python/sglang/kernels/ops/gemm/__init__.py:125<br>apply_unquantized_linear @ python/sglang/multimodal_gen/runtime/layers/linear.py:134<br>_topk_tile_lists @ python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn_h3.py:228 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | Matches an open upstream path (34.8% related GPU time). Open SGLang PR replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RMSNorm + RoPE | Confirmed | 56.71 ms | 1.7% | void sglang::fused_qknorm_rope_warp<128l, 96l, true, true, __nv_bfloat16, __nv_bfloat16, true, false, false, long, false> (1.7%) | forward @ python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1041 | python/sglang/jit_kernel/fused_qknorm_rope.py<br>python/sglang/srt/models/qwen3_moe.py | `Fused QK RMSNorm + RoPE` is present in this trace (1.7% related GPU time). SGLang has a fused QK-norm plus RoPE kernel family. |
