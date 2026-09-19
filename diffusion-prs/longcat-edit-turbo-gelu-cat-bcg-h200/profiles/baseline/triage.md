Diagnostic cumulative GPU times. The automated nvjet-to-FP8 suggestion is rejected: these GEMMs are BF16.

{
  "source": "/campaign/artifacts/longcat-image-edit-turbo/baseline-profile/traces/ea1b3284-ebf6-4e23-b68d-f74ce53c2a5b-full_stages-global-rank0.trace.json.gz",
  "source_events": 338404,
  "slice": "/campaign/artifacts/longcat-image-edit-turbo/baseline-profile/traces/denoise-step2.trace.json.gz",
  "slice_events": 29213,
  "model_calls": 8,
  "boundary": "third to fourth model start, with native stage synchronization enabled",
  "start_us": 6880752382121.489,
  "end_us": 6880752537724.467,
  "crossing_kernels": 0,
  "window_ms": 155.6029775390625,
  "gpu_union_ms": 151.6625205078125,
  "uncovered_interval_ms": 3.9404570312499914,
  "uncovered_interval_pct": 2.532378938738997,
  "kernel_count": 699,
  "launch_api_count": 699,
  "top_kernels": [
    {
      "name": "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<176>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, false, false, false, false, false, true, true, false, false, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<176> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, false, false, false, false>, flash::StaticPersistentTileScheduler<false> > > >(flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<176>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, false, false, false, false, false, true, true, false, false, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<176> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, false, false, false, false>, flash::StaticPersistentTileScheduler<false> > >::Params)",
      "count": 30,
      "total_ms": 47.875426000000004
    },
    {
      "name": "nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_bias_TNT",
      "count": 80,
      "total_ms": 44.985700999999985
    },
    {
      "name": "nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN",
      "count": 80,
      "total_ms": 37.613888999999986
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<8, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)",
      "count": 40,
      "total_ms": 4.411873999999999
    },
    {
      "name": "void sglang::fused_qknorm_rope_warp<128l, 128l, false, true, __nv_bfloat16, float, true, false, true, long, false>(std::conditional<false, sglang::QKNormRopePackKVParams, std::conditional<false, sglang::QKNormRopeOutOfPlaceParams, sglang::QKNormRopeParams>::type>::type)",
      "count": 40,
      "total_ms": 3.440806
    },
    {
      "name": "void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 3, 128, 1, 16, 8>(char*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)",
      "count": 20,
      "total_ms": 2.8097950000000003
    },
    {
      "name": "_layernorm_modulate_kernel",
      "count": 60,
      "total_ms": 2.32886
    },
    {
      "name": "nvjet_sm90_tst_192x112_64x5_1x2_h_bz_coopB_bias_TNT",
      "count": 51,
      "total_ms": 2.217023
    },
    {
      "name": "void sglang::residual_gate_add::(anonymous namespace)::residual_gate_add_broadcast_kernel<__nv_bfloat16, 8>(__nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16 const*, long, long)",
      "count": 60,
      "total_ms": 1.646176
    },
    {
      "name": "void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1, 16, 8>(char*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)",
      "count": 51,
      "total_ms": 1.4808620000000006
    },
    {
      "name": "nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_bias_TNT",
      "count": 10,
      "total_ms": 0.9606399999999999
    },
    {
      "name": "nvjet_sm90_tst_192x8_64x8_4x1_v_bz_bias_TNT",
      "count": 20,
      "total_ms": 0.7372780000000002
    },
    {
      "name": "nvjet_sm90_tst_192x8_64x8_4x1_v_bz_splitK_bias_TNT",
      "count": 20,
      "total_ms": 0.44509000000000004
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<8, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)",
      "count": 42,
      "total_ms": 0.08569500000000001
    },
    {
      "name": "void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1})",
      "count": 1,
      "total_ms": 0.073697
    },
    {
      "name": "void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})",
      "count": 1,
      "total_ms": 0.072544
    },
    {
      "name": "void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, true, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)",
      "count": 21,
      "total_ms": 0.053024
    },
    {
      "name": "void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float, false>(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, float*, float*, c10::BFloat16*)",
      "count": 1,
      "total_ms": 0.045344
    },
    {
      "name": "void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)",
      "count": 8,
      "total_ms": 0.026465
    },
    {
      "name": "nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN",
      "count": 1,
      "total_ms": 0.025632000000000002
    },
    {
      "name": "nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT",
      "count": 2,
      "total_ms": 0.020543
    },
    {
      "name": "nvjet_sm90_tst_64x88_64x11_1x4_h_bz_bias_TNT",
      "count": 1,
      "total_ms": 0.016159
    },
    {
      "name": "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1} const&)::{lambda(int)#1})",
      "count": 6,
      "total_ms": 0.016128000000000003
    },
    {
      "name": "void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > const&)::{lambda(int)#1})",
      "count": 3,
      "total_ms": 0.014047999999999998
    },
    {
      "name": "void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 2, 128, 1, 16, 4>(char*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)",
      "count": 5,
      "total_ms": 0.014015999999999999
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::sin_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::sin_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "count": 3,
      "total_ms": 0.009951999999999999
    },
    {
      "name": "nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_bias_TNT",
      "count": 1,
      "total_ms": 0.009504
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::cos_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::cos_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "count": 3,
      "total_ms": 0.009312
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 2ul>)",
      "count": 6,
      "total_ms": 0.006496
    },
    {
      "name": "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::pow_tensor_tensor_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(double, double)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::pow_tensor_tensor_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(double, double)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::pow_tensor_tensor_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(double, double)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::pow_tensor_tensor_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(double, double)#1} const&)::{lambda(int)#1})",
      "count": 3,
      "total_ms": 0.005824
    },
    {
      "name": "void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})",
      "count": 1,
      "total_ms": 0.005728
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::reciprocal_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::reciprocal_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "count": 3,
      "total_ms": 0.004128
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)",
      "count": 3,
      "total_ms": 0.003968
    },
    {
      "name": "void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<2>, at::native::memory::StoreWithCast<1> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<2>, at::native::memory::StoreWithCast<1>)",
      "count": 1,
      "total_ms": 0.003936
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 2ul>)",
      "count": 3,
      "total_ms": 0.003264
    }
  ],
  "note": "Uncovered GPU intervals suggest a launch-overhead opportunity; they are not a promise of recoverable E2E time."
}
Triage View
Mode: single-trace
Framework: SGLang
Input traces: /campaign/artifacts/longcat-image-edit-turbo/baseline-profile/traces/denoise-step2.trace.json.gz

Kernel Table
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<176>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, false, false, false, false, false, true, true, false, false, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<176> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, false, false, false, false>, flash::StaticPersistentTileScheduler<false> > > > | gemm | 47.88 ms | 31.6% | 30 | python/sglang/kernels/ops/attention/flash_attention_v3.py:19 _call_fa3_kernel | sgl_kernel::fwd |
| nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_bias_TNT | gemm | 44.99 ms | 29.7% | 80 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::addmm |
| nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN | gemm | 37.61 ms | 24.8% | 80 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::addmm |
| void at::native::vectorized_elementwise_kernel<8, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> > | activation | 4.41 ms | 2.9% | 40 | python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py:637 forward (site share 67%)<br>python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py:301 forward (site share 33%) | aten::gelu<br>aten::gelu |
| void sglang::fused_qknorm_rope_warp<128l, 128l, false, true, __nv_bfloat16, float, true, false, true, long, false> | rope | 3.44 ms | 2.3% | 40 | python/sglang/multimodal_gen/runtime/layers/layernorm.py:1062 apply_qk_norm_rope | sglang::fused_inplace_qknorm_rope |
| void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 3, 128, 1, 16, 8> | memory | 2.81 ms | 1.9% | 20 | python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py:637 forward | aten::cat |
| _layernorm_modulate_kernel | norm | 2.33 ms | 1.5% | 60 | python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py:69 _longcat_norm_modulate | sglang::triton_fused_layernorm_modulate |
| nvjet_sm90_tst_192x112_64x5_1x2_h_bz_coopB_bias_TNT | gemm | 2.22 ms | 1.5% | 51 | python/sglang/multimodal_gen/runtime/layers/linear.py:134 apply_unquantized_linear | aten::addmm |
| void sglang::residual_gate_add::(anonymous namespace)::residual_gate_add_broadcast_kernel<__nv_bfloat16, 8> | communication | 1.65 ms | 1.1% | 60 | python/sglang/kernels/kda_kernels/residual_gate_add_jit.py:155 residual_gate_add_cuda | sglang::diffusion_residual_gate_add |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Confirmed | 87.20 ms | 57.5% | nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_bias_TNT (29.7%)<br>nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN (24.8%)<br>nvjet_sm90_tst_192x112_64x5_1x2_h_bz_coopB_bias_TNT (1.5%) | apply_unquantized_linear @ python/sglang/multimodal_gen/runtime/layers/linear.py:134<br>forward @ python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py:113<br>forward @ python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py:132<br>forward @ python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py:917 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | Matches an open upstream path (57.5% related GPU time). Open SGLang PR replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RMSNorm + RoPE | Confirmed | 3.44 ms | 2.3% | void sglang::fused_qknorm_rope_warp<128l, 128l, false, true, __nv_bfloat16, float, true, false, true, long, false> (2.3%) | apply_qk_norm_rope @ python/sglang/multimodal_gen/runtime/layers/layernorm.py:1062 | python/sglang/jit_kernel/fused_qknorm_rope.py<br>python/sglang/srt/models/qwen3_moe.py | `Fused QK RMSNorm + RoPE` is present in this trace (2.3% related GPU time). SGLang has a fused QK-norm plus RoPE kernel family. |
