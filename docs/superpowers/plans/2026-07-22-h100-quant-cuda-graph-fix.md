# H100 Masked Quant CUDA Graph Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the H100 invalid-host-pointer failure when the unified masked `per_token_group_quant` path feeds DeepGEMM during compiled decode CUDA graph capture.

**Architecture:** Reproduce the production call boundary on H100, identify the exact tensor whose runtime storage/device contract is invalid, and correct the allocation/custom-op boundary without disabling the unified kernel. Add a focused graph-capture regression test, then validate the complete DeepEP TBO server test.

**Tech Stack:** Python, PyTorch custom operators and `torch.compile`, CUDA graphs, TVM-FFI, DeepGEMM, SGLang JIT kernels, pytest/unittest, 4x NVIDIA H100.

---

## File Map

- Modify `python/sglang/kernels/ops/quantization/per_token_group_quant.py` only if the returned-output/custom-op contract is the failing boundary.
- Modify `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` only if caller-owned output allocation or lifetime is the failing boundary.
- Modify `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` only for a permanent, narrowly scoped device-contract validation justified by the reproducer.
- Modify `test/registered/jit/test_per_token_group_quant.py` for the focused compiled and CUDA-graph regression.
- Use `test/registered/ep/test_deepep_small.py` unchanged as the end-to-end regression.

### Task 1: Prepare and Baseline the H100 Environment

- [ ] Assign the available Radix H100 node for a 4-hour validation window.

Run:

```bash
export PATH="$HOME/.local/bin:$PATH"
export RADIX_API=https://nodes.sglang.io
radix assign host-85-234-79-62 --gpus 8 --duration 4h
radix machines mine
```

Expected: `host-85-234-79-62` is assigned and reports 8 NVIDIA H100 80GB GPUs.

- [ ] Inspect the host before using it.

Run through `radix shell host-85-234-79-62`:

```bash
hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader
df -hT / /data /data01 /data02 2>/dev/null || df -hT
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

Expected: at least GPUs 0-3 are idle, Docker is usable, and a writable disk has enough space for the repository and model cache.

- [ ] Create or reuse the personal `sglang_bbuf` container and sync the clean worktree at commit `02e8a6f676` into `/tmp/sglang_h100_quant_fix`.

- [ ] Install the checkout editable and record package versions.

Run inside the container:

```bash
cd /tmp/sglang_h100_quant_fix
pip install -e 'python[all]'
python - <<'PY'
import deep_gemm, torch, tvm_ffi
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('deep_gemm', getattr(deep_gemm, '__version__', 'unknown'))
print('tvm_ffi', getattr(tvm_ffi, '__version__', 'unknown'))
print('gpu', torch.cuda.get_device_name(0))
PY
```

Expected: CUDA is available and the GPU is H100.

### Task 2: Reproduce and Identify the Invalid Tensor

- [ ] Run the failing server test unchanged on four H100 GPUs.

Run:

```bash
cd /tmp/sglang_h100_quant_fix
CUDA_VISIBLE_DEVICES=0,1,2,3 python test/registered/ep/test_deepep_small.py \
  TestTBOWithTPAttn 2>&1 | tee /tmp/h100_tbo_parent.log
```

Expected on the parent commit: startup fails during decode CUDA graph capture with `The specified pointer resides on host memory`.

- [ ] Add temporary boundary diagnostics immediately before the second `grouped_gemm_nt_f8f8bf16_masked` call. For each tensor (`down_input`, `down_input_scale`, `w2_weight`, `w2_scale`, `down_output`, `masked_m`, and overlap signal), print type, `device`, `is_cuda`, dtype, shape, stride, storage offset, `data_ptr`, and whether it is a `torch._subclasses.fake_tensor.FakeTensor`.

- [ ] Repeat the failing test and save `/tmp/h100_tbo_diagnostic.log`.

Expected: the log names exactly one argument with host/Fake/invalid storage semantics at the call that raises.

- [ ] Remove the temporary prints after recording the result. Preserve the diagnostic evidence in the PR description rather than production code.

### Task 3: Add the Focused Regression Test

- [ ] Add a test to `test/registered/jit/test_per_token_group_quant.py` that constructs the masked fused-SiLU shape used by the H100 MoE path, allocates caller-owned CUDA outputs, invokes the registered quant op, and consumes both outputs after `torch.compile` and CUDA graph capture.

The assertions must verify:

```python
assert output_q.is_cuda
assert output_s.is_cuda
assert output_q.device == input.device
assert output_s.device == input.device
assert not isinstance(output_q, FakeTensor)
assert not isinstance(output_s, FakeTensor)
```

It must also compare replayed output against eager output and pass the quantized tensors into the smallest available DeepGEMM masked call that reproduces the original TVM-FFI conversion.

- [ ] Run the focused test on the parent implementation.

Run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q \
  test/registered/jit/test_per_token_group_quant.py \
  -k 'masked and compile and cuda_graph' -vv
```

Expected: FAIL with the same invalid-host-pointer signature or with the diagnosed device/alias contract violation.

- [ ] Commit the reproducing test separately.

```bash
git add test/registered/jit/test_per_token_group_quant.py
git commit -m 'test: reproduce H100 masked quant graph failure'
```

### Task 4: Correct the Quant-to-DeepGEMM Contract

- [ ] Keep allocation outside the opaque mutating custom op: allocate `output_q` and `output_s` on `input.device` in the caller-visible `per_token_group_quant` wrapper, then pass them explicitly to `_per_token_group_quant_custom_op`.

- [ ] Ensure the registered custom op remains in-place-only with:

```python
@register_custom_op(
    op_name="per_token_group_quant",
    mutates_args=["output_q", "output_s"],
)
```

and returns `None`; do not let fake execution manufacture returned output tensors.

- [ ] If diagnostics show the registry wrapper is the graph break, resolve the JIT implementation once outside the compiled forward and call the registered in-place op directly. Do not decorate the whole quant path with `torch.compiler.disable`.

- [ ] If diagnostics instead identify a transformed scale view, retain the owning allocation and create the exact column-major view before entering capture; assert that its base storage remains CUDA-backed through the DeepGEMM call.

- [ ] Run the focused regression test.

Expected: PASS in eager, compiled, capture, and compiled-capture modes on H100.

- [ ] Run all masked quant tests on one H100.

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q \
  test/registered/jit/test_per_token_group_quant.py \
  -k masked -vv
```

Expected: all selected tests pass.

- [ ] Commit the minimal implementation.

```bash
git add python/sglang/kernels/ops/quantization/per_token_group_quant.py \
  python/sglang/srt/layers/moe/moe_runner/deep_gemm.py \
  python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py
git commit -m 'fix: preserve CUDA tensors across masked quant graphs'
```

Only add paths that actually changed.

### Task 5: Validate the Full H100 Failure Path

- [ ] Run `TestTBOWithTPAttn` on four H100 GPUs with the fix.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python test/registered/ep/test_deepep_small.py \
  TestTBOWithTPAttn 2>&1 | tee /tmp/h100_tbo_fixed.log
```

Expected: server completes CUDA graph capture through batch size 128 and the test passes without an invalid host pointer.

- [ ] Run the full DeepEP-small test file if the focused class passes.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python test/registered/ep/test_deepep_small.py \
  2>&1 | tee /tmp/h100_deepep_small_fixed.log
```

Expected: no new failures attributable to the patch.

- [ ] Run repository checks on the changed files.

```bash
pre-commit run --files \
  python/sglang/kernels/ops/quantization/per_token_group_quant.py \
  python/sglang/srt/layers/moe/moe_runner/deep_gemm.py \
  python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py \
  test/registered/jit/test_per_token_group_quant.py
git diff --check origin/main...HEAD
```

Expected: PASS.

### Task 6: Publish the Fix

- [ ] Review scope and history.

```bash
git status --short --branch
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

Expected: only the design, plan, focused regression, and minimal fix are present.

- [ ] Push the branch to the user's fork.

```bash
git push -u fork agent/fix-h100-quant-cuda-graph
```

- [ ] Open a draft PR against `sgl-project/sglang:main` with title:

```text
[Fix] Preserve CUDA tensor storage across masked quant graph capture
```

The PR body must include the exact offending tensor, why #30924 exposed the issue, why it appears in `TestTBOWithTPAttn`, the before/after H100 logs, focused test results, full test results, and remaining B200 validation risk.

- [ ] Return the draft PR URL and validation summary to the user.
