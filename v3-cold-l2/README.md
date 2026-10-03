# Diffusion cleanup: cold-L2 validation

PR [#42391](https://github.com/sgl-project/sglang/pull/42391), final tested head **`290c07882532e4be00f670865087c53da5ce4efa`**. Baseline **`5916999afbc595de62dffca077fd8249634b4e74`**. A dedicated B200 was used for sequential baseline/candidate runs. Package/driver metadata and source verification are archived with the results.

## Findings and fixes

Two actual regressions were found and fixed during this validation:

1. **CPU fallback poisoned Wan's GPU fusion gate.** With CUDA available, a CPU request entered a CUDA launcher. Its exception permanently disabled the process-wide fusion gate; later GPU requests returned non-contiguous eager slices and incurred downstream copies. Existing Wan tests pass 3/3 on baseline, fail 1/3 on `f99140e5f1`, and pass 3/3 after `88882677ee`. Analogous model entry points now choose CPU/CUDA from the actual input device. Shape, dtype and layout relationships remain launcher validation.
2. **Ideogram scanned the same inputs twice.** Its model predicate remained after validation moved into the RoPE launcher, adding about 5.6 us (9–12%) to the measured eager helper. `290c078825` removes that duplicate scan and the flattening views. The launcher accepts the original contiguous `(B,S,1,R)` cos/sin tables as well as its existing flat rows. The model retains its unsupported-layout fallback. Corrected helper measurements no longer reproduce the regression; both candidate repetitions are faster than both baseline repetitions for each measured shape.

No device arithmetic or launch tuning changed in either fix. The source-body comparison covers **36 unchanged retained Triton/CUDA definitions** (normalizing only Helios' added FP32 type assertion); five removed definitions are dead residual-gate alternatives. No input-legality tests were added.

## Correctness

| Check | Result |
| --- | --- |
| Baseline: full diffusion kernel directory + namespace | 3002 passed, 89 skipped, 4 failed; 40 subtests passed |
| Final `290c078825`: same kernel suites | **3011 passed, 89 skipped, the same 4 failures; 40 subtests passed** |
| Final: existing Ideogram, Ernie RoPE/GeGLU and Wan model tests | **11 passed** |
| Final: Ideogram fullgraph compile + CUDA Graph replay after replacing inputs | **Both tested shapes passed**, including B=2 |
| Final: 84 cold-L2 workloads, four GPU-event runs and four eager runs | **All output bytes, shapes and strides identical to baseline** |
| Final: 7 additional model-helper workloads, four eager runs | **All outputs identical**; Ernie/Ideogram numerical fusion gates verified |
| Full local model unit suite at CPU-fix revision `88882677ee` | 3787 passed, 133 skipped, 3 xfailed, 351 subtests passed; 1 baseline failure |
| Final `290c078825`: full model-unit CI | [Passed](https://github.com/sgl-project/sglang/actions/runs/37139995184/job/111252398758) |
| Full model CI at `88882677ee` | [3787 passed, 134 skipped, 3 xfailed, 351 subtests passed](https://github.com/sgl-project/sglang/actions/runs/37137003091/job/111243696751) |

The four kernel failures are `test_ernie_qknorm_rope_is_bit_exact` and the three split-BF16-rounding/full-width-NeoX/packed-KV cases in `test_rope.py`; baseline reproduces them in this environment. The local model-suite failure is `TestSubBlockNumerics.test_ragged_tail_reproduces_dense` (NaN cosine). Running its entire file separately on baseline and `88882677ee` gives the same **1 failed, 24 passed, 3 skipped**, with 19 subtests passed. This is not counted as a passing check.

The full model-suite result predates the final two-file Ideogram fix. Final-head coverage for those files consists of the existing 11 model tests, the full kernel suites, actual helper benchmarks and explicit compile/graph replay checks; the final-head full model-unit CI has now also passed. The source manifests verify 1,612 kernel/model/test files for each full benchmark tree.

## Cold-L2 method

Reported L2 is **132,644,864 bytes (126.5 MiB)**. Before every measured invocation, a `uint8.add_(1)` reads/writes **663,224,320 bytes (632.5 MiB, 5x L2)**. Input restoration and eviction are outside the timed interval. A 16 MiB cache probe measures about 8.19 us hot versus 12.23 us cold (1.49x). GPU clocks were not locked.

The previous v2 benchmark capped input rotation at 100 buffers, so small cases did not necessarily exceed L2. **Its historical rotating-input results must not be presented as per-invocation cold-L2 measurements.**

Three methods are retained here:

- **Ordinary CUDA events, primary GPU measurement:** restore inputs; queue a GPU delay to let the CPU enqueue the complete interval; evict L2; start event; one operator invocation; end event. Delay and eviction are excluded. Ten warmups and 100 measured calls.
- **Eager wall time:** restore inputs; evict L2; synchronize; CPU timer; one operator; synchronize; CPU timer. This includes Python/C++ dispatch, GPU execution and completion synchronization. Ten warmups and 100 calls.
- **Diagnostic CUDA Graph events:** restore inputs and evict inside a captured graph, then external start/end events around one operator. Ten graph warmups and 100 replays. This avoids Python dispatch in the timed region but produced capture-dependent event overhead on tiny operations, as detailed below.

Fixed seed, separate processes in **A/B/B/A** order, no competing validation workload on the allocated GPU. Compare the means of the two run medians per arm; all individual samples and run medians are preserved. Empty timing controls (~3 us for GPU events, ~5.3 us for eager synchronization) are reported and **not subtracted**. Packed views are reconstructed from their parent buffers inside model wrappers so cloning does not accidentally turn them contiguous.

## Final performance results

| Measurement | Cases | Candidate latency change |
| --- | ---: | --- |
| GPU events: direct kernel APIs / Helios attention entry | 72 | **-4.62% to +0.46%; geometric mean -0.19%** |
| GPU events: Qwen/GLM/Ernie/Wan model helpers | 12 | -84.99% to 0.00% |
| Eager wall time: those model helpers | 12 | -78.25% to **+2.08% (maximum +0.82 us)** |
| Eager wall time: additional Ernie/Ideogram/Sensenova helpers | 7 | -7.33% to -0.91% |

The large Qwen packed-input improvement comes from reaching the existing native stride-aware QKNorm/RoPE implementation. This is an operator/helper result, not an end-to-end throughput claim. The small negative GPU deltas on unchanged kernels are not attributed to an optimization.

**Host-cost caveat:** invoking several bare Triton launchers now includes validation previously performed by their callers. Against bare baseline launchers, those eager microbenchmarks add roughly **2–6 us** (up to 15.58%). The actual model-helper comparisons include the old caller checks and the new launcher checks. The Ideogram duplicate-check regression was found this way and fixed. This PR should therefore not be described as zero overhead for every direct API call or every platform.

Detailed tables: [GPU intervals](results/stream/comparison.csv), [eager intervals](results/eager/comparison.csv), [additional model helpers](results/more-models/comparison.csv).

## Investigating timing outliers

At `88882677ee`, external graph-event intervals for some tiny kernels alternated between about 6 and 8 us, initially suggesting increases as large as 32%. Recapturing the same operation in the same revision also produces both ranges. Ordinary-event runs do not reproduce those large increases. Nsight Systems kernel medians for the four investigated operations are:

| Kernel | Baseline | Candidate `88882677ee` |
| --- | ---: | ---: |
| causal-conv cat/pad | 2400 ns | 2400 ns |
| complex RoPE | 1952 ns | 1952 ns |
| QKNorm complex RoPE | 2048 ns | 2080 ns |
| SiLU-mul | 1888 ns | 1888 ns |

Two FP16 residual cases also showed inconsistent ordinary-event differences in the earlier run. Their device definitions are unchanged; the final event repetitions do not reproduce an increase. Final-head Nsight profiles give **5216 -> 5216 ns** for the small grid and **31648 -> 31680 ns** for the large grid. Identical grid/block dimensions and the original broadcast CUDA kernel are confirmed. These observations support measurement/scheduling variation rather than a reproducible device-kernel regression. Every diagnostic run is retained; the earlier unfavorable measurements were not discarded.

Nsight summaries include 121 calls per operator/grid (110 cold graph replays plus first-call/warmup activity); medians are dominated by cold calls. Raw `.nsys-rep` files are retained in the local report directory; public evidence includes CSV exports and profiling commands.

## Evidence layout and reproduction

- `results/`: final head `290c078825`; `manifest-reviewed.json` and `source-manifest.json` identify its sources.
- `before-ideogram-fix/`: `88882677ee` measurements, full model tests, and the timing investigation. Its `candidate-wan.log` is the explicit earlier `f99140e5f1` reproduction; `candidate-final-wan.log` is the fixed version.
- `manifest-final.json`: source manifest for `88882677ee`; `manifest-prefixed.json`: `f99140e5f1`. These historical filenames follow the original run directories.
- `run-reviewed.sh` records the final kernel tests, helper tests, A/B runs and residual profiles. `run-compile-check.sh` records the added compile/graph experiment. Scripts use `/scratch/cold-l2/{baseline,candidate-reviewed}`; populate those directories with the exact Git revisions and use the archived dependency versions.
- Benchmarks live outside the checkout. Set `PYTHONPATH=$PWD/python`, `ARM=baseline` or `candidate`, and `OUTPUT=/path/to/result.json`, then run the appropriate `bench_cold_*.py` in each checkout. Each script uses 100 measurements and the same 5x-L2 eviction.

This validates measured B200 shapes/layouts and the listed model helpers. It is not a universal guarantee for arbitrary inputs, all architectures, multi-GPU execution or full checkpoint denoising throughput. [Final-head lint](https://github.com/sgl-project/sglang/actions/runs/37139994854) passed; [fresh final-head CI](https://github.com/sgl-project/sglang/actions/runs/37139995184) was triggered and its snapshot is recorded separately.


Final-head CI snapshot: full model unit tests, B200 diffusion, RTX 5090 diffusion and component accuracy passed. The 4-GPU JIT job failed in the unchanged DSV4.1 sparse-indexer test `test_prefill_tail_rebuilds_the_last_rows` (overlap 501, required 501.76, row 14). Both its test file and the attention subtree are unchanged by this PR. This particular CI failure has not been reproduced on baseline. A job-only rerun was attempted, but GitHub refused while the parent workflow was still running (HTTP 403). **CI is not all green**; see [the recorded snapshot](ci-summary.json).
