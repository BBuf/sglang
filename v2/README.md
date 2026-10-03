# Expanded diffusion kernel cleanup

Baseline: `5916999afbc595de62dffca077fd8249634b4e74`. GPU-tested runtime snapshot: `b10ade0673645d8515b7b74abac732f71486c6b2`. [Implementation PR](https://github.com/sgl-project/sglang/pull/42391).

This revision removes 17 tensor capability predicates and narrows five selectors to scalar algorithm configuration. Native CUDA host launchers validate original tensor shapes and relationships. Triton launchers own their validation. Numerical verification, platform choices and layout-dependent algorithms remain. The [design and remaining audit](design-review.md) distinguish what changed from specialized paths still requiring further work.

Current PR head `f99140e5f1` removes input-validation-only tests, capability assertions and unused test imports. Runtime code is unchanged. The GPU test counts and manifest below describe the archived snapshot before these test deletions; they are not new-run counts for the reduced test suite.

## Source and hardware

Dedicated 1-GPU NVIDIA B200 devbox, `lmsysorg/sglang:dev`, PyTorch 2.13.0+cu130, Triton 3.7.1. CUDA_VISIBLE_DEVICES=0, MAX_JOBS=8. No concurrent GPU test or benchmark jobs during timing.

The [source manifest](source-manifest.json) records all 63 changed file hashes and the device-source comparison: 36 retained kernel/helper definitions unchanged, five dead residual-gate definitions deleted. The comparison normalizes only Helios' assertion allowing FP32. No active device math or launch tuning was changed. The final [remote verification](results/source-verification.json) verifies all changed files against this manifest.

The 72-case graph benchmark was taken at `22a92833a9`. The final commit only reuses validated metadata in three normalization launchers and updates their documentation; it does not alter device code or launch parameters. The final model-dispatch benchmark and final full correctness run use the final runtime sources.

## Correctness

See [validation-summary.json](validation-summary.json) and raw logs in [results](results/). Baseline: 3,002 passed, 89 skipped, four failures, 40 subtests passed. The final candidate passed **3,054 tests and 40 subtests**, skipped 89, and reproduced exactly those four failures. [Final full log](results/candidate-final-full.log). The four known failures are:

- `test_model_fast_paths.py::test_ernie_qknorm_rope_is_bit_exact`
- `test_rope.py::test_qknorm_rope_preserves_split_bf16_rounding`
- `test_rope.py::test_qknorm_rope_preserves_full_width_neox_cache`
- `test_rope.py::test_qknorm_rope_pack_kv_preserves_split_bf16_rounding`

These are in the unchanged generic QKNorm/RoPE rounding path on this environment. Ernie's output equals its eager reference; that test fails because its fusion gate correctly declines to verify the native path. They are not hidden or marked as passing.

Coverage includes native rejection before mutation, same-numel/different-shape inputs, dtype/device/layout/alignment errors, compiled fake-output layouts and packed Qwen model calls with native and Triton alternatives enabled/disabled. No full-checkpoint denoising, multi-GPU TP/SP or AMD/XPU execution was performed. H200 evidence in the parent directory covers only the original, narrower commit.

## Performance

[72-case harness](bench_cleanup.py), [analysis](analyze_benchmarks.py), [summary](benchmark-summary.json), and four `bench-*.json` files in [results](results/).

A/B/B/A process order, fixed seed 20261003, repository CUDA Graph marker, 10 warmups and 100 replays with rotating cloned buffers. Output bytes, shapes and strides match in all four runs. Packed Qwen workloads pass the projection buffer into the marker and create views inside the measured wrapper, preserving the intended strides when the marker clones its inputs.

The 68 direct kernels range from **-0.90% to +1.57%** graph latency change, geometric mean **+0.008%**; the largest positive difference is about 0.03 us on a 2 us kernel. The four additional Qwen model-helper cases show unchanged contiguous graph latency (within 0.02%). Packed projection views reach the existing stride-aware CUDA path: **23.65 → 2.49 us** at S=17 and **227.87 → 35.36 us** at S=4096. These are operator/helper measurements, not whole-model throughput claims.

Direct-call eager measurements omit the old model-side predicates, so they cannot alone compare total launch overhead. The [model-dispatch harness](bench_model_dispatch.py) measures actual GLM/Ernie wrappers with the fusion gate verified and active: 20 warmups, median of five rounds of 1,000 calls, synchronize at round boundaries, A/B/B/A process order. Eager time includes host dispatch, allocation and device work. All six outputs are byte-identical and all gates are verified in all runs.

| Model helper / sequence | Baseline us | Final candidate us | Change |
| --- | ---: | ---: | ---: |
| `glm_ln/17` | 33.168 | 31.753 | -4.27% |
| `glm_qk_ln/17` | 26.965 | 27.161 | +0.72% |
| `ernie_rms/17` | 22.959 | 22.545 | -1.80% |
| `glm_ln/4096` | 32.870 | 31.575 | -3.94% |
| `glm_qk_ln/4096` | 40.234 | 40.184 | -0.13% |
| `ernie_rms/4096` | 22.988 | 22.722 | -1.16% |

The first expanded version added roughly 1 us to these wrappers by repeatedly reading tensor metadata. The final commit computes shape/row layout once during validation and reuses it for launch. [Before-fix measurements](model-dispatch-before-summary.json) and [raw inputs/results](results/model-before/) are retained. The final six-case range is **-4.27% to +0.72%**; these measurements show no material eager regression, not a universal speed guarantee.

## Reproduce

Use the same runtime versions and separate checkouts for baseline/candidate. Set EVIDENCE to this `v2` directory. From either source checkout:

```bash
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8
export PYTHONPATH="$PWD/python"
python3 -m pytest -q test/registered/kernels/ops/diffusion test/registered/unit/kernels/test_kernels_namespace.py
ARM=baseline OUTPUT=bench-A1.json python3 "$EVIDENCE/bench_cleanup.py"
OUTPUT=model-A1.json python3 "$EVIDENCE/bench_model_dispatch.py"
# Repeat in candidate, candidate, baseline order as B1, B2, A2.
python3 "$EVIDENCE/analyze_model_dispatch.py" /path/to/results
python3 "$EVIDENCE/verify_source_manifest.py" /path/to/final-candidate "$EVIDENCE/source-manifest.json"
```

The `run-*.sh` files record the original container paths. Adapt `/scratch/cleanup-v2` to the replay directory. Evidence is kept outside the implementation diff.
