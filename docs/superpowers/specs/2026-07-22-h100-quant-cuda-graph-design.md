# H100 Masked Quant CUDA Graph Compatibility Design

## Problem

`test/registered/ep/test_deepep_small.py::TestTBOWithTPAttn` fails while
capturing the decode CUDA graph on an H100 system. The first actionable error
is raised by DeepGEMM while converting a TVM-FFI tensor view to a PyTorch
tensor:

```text
RuntimeError: The specified pointer resides on host memory and is not
registered with any CUDA device.
```

The failure occurs after the plain-SiLU masked MoE path calls the unified
`per_token_group_quant` operator introduced by PR #30924 and passes its results
to `grouped_gemm_nt_f8f8bf16_masked`. The later process exit code `-9` is
cleanup after the scheduler exception, not the root cause. Available GPU memory
at the failure is sufficient, so this is not treated as an OOM.

## Goal

Identify which tensor crosses the quant-to-DeepGEMM boundary with invalid
device or pointer semantics under `torch.compile` and CUDA graph capture, fix
the underlying custom-op or tensor-lifetime contract without disabling the new
kernel on H100, and add regression coverage.

## Non-goals

- Do not fall back to the pre-#30924 Triton implementation on H100.
- Do not disable decode CUDA graphs or `torch.compile` as the final fix.
- Do not reduce CUDA graph batch sizes merely to avoid the failing shape.
- Do not refactor unrelated MoE, DeepEP, or kernel-registry code.

## Investigation Strategy

Use a clean H100 environment matching CI as closely as practical. Reproduce the
full `TestTBOWithTPAttn` startup failure first, then reduce it to the smallest
case that preserves the invalid-pointer behavior.

At the call to `grouped_gemm_nt_f8f8bf16_masked`, inspect every tensor argument:

- `down_input`
- `down_input_scale`
- `w2_weight` and `w2_scale`
- `down_output`
- `masked_m`
- any overlap signal tensor

For each tensor, capture its Python type, fake-tensor status, device, dtype,
shape, stride, storage offset, data pointer, and CUDA registration behavior.
Run the reduced case in four modes: eager, `torch.compile`, CUDA graph capture,
and `torch.compile` inside CUDA graph capture. This separates a kernel error
from a graph-boundary or fake-tensor error.

Temporary diagnostics must either be removed before the final commit or be
converted into narrowly useful validation with negligible production cost.

## Fix Selection

Choose the smallest correction justified by the reproduction:

1. If an allocated quant output becomes a FakeTensor or host-backed tensor,
   correct the custom-op schema and fake implementation so tracing preserves
   CUDA device and alias/mutation semantics.
2. If output allocation is incorrectly captured, allocate caller-owned CUDA
   outputs outside the opaque custom op and keep only the in-place quant launch
   inside the registered op.
3. If a view or scale-layout transformation loses valid storage/device
   metadata, correct that transformation while preserving the layout expected
   by DeepGEMM.
4. If TVM-FFI conversion observes a dead or recycled allocation, retain the
   owning tensor for the complete DeepGEMM call and correct the lifetime at the
   narrowest boundary.

The final patch must preserve the unified kernel on both SM90 and SM100.

## Tests

Add a focused regression test that exercises the masked fused-SiLU quant path
feeding DeepGEMM on H100 under the mode that originally failed. It must fail on
the parent commit with the invalid host-pointer error and pass after the fix.

Validation layers:

1. Focused quant/custom-op unit test in eager and compiled modes.
2. Focused CUDA graph capture reproducer using H100.
3. Existing `per_token_group_quant` registered tests relevant to masked FP8.
4. Full `TestTBOWithTPAttn` server startup and test on 4x H100.
5. Static checks and the directly affected CPU tests, where applicable.

## Delivery

Work on a branch based on current `origin/main`. Commit only the focused fix,
regression tests, and this design/implementation documentation. Push to the
user's fork and open a draft PR against `sgl-project/sglang:main` containing the
root cause, before/after failure evidence, H100 validation commands, and any
remaining cross-architecture risk.
