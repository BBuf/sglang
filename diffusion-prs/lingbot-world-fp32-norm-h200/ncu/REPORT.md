# LingBot World FP32 normalization on H200

## Workload and method

Final candidate `89b3700dbc9146727cbb368b6859f5a18e79ae66`, based on `dd83b54611897a5f80f4df59e69756bd2fb4b8ab`. NVIDIA H200 on assigned GPU 0; Nsight Compute 2025.3.1, Torch 2.13.0+cu130 at `cf30153c4c131c8164ee7798e5022d810682e2cb`, Triton 3.7.1. Native benchmark/profile and NCU requests run separately under the campaign GPU lock.

Both sites use contiguous BF16 [1,4680,5120] input/output, FP32 [1,5120] affine tensors, eps1e-6, seed 1729. Norm1's baseline is BF16->FP32, native LayerNorm, separate (1+scale), multiplication, shift addition, BF16 store. Cross norm is BF16->FP32, native affine LayerNorm, BF16 store. The fused helper retains ATen's Welford reduction order, uses explicit FMA only for affine, and disables implicit FMA for modulation. The harness compares raw int16 bits before profiling, warms 10 times and brackets one whole site with cudaProfilerStart/Stop.

Every provider has a full report with PM sampling and a separate source/SourceCounters report. Report replay changes cache conditions; these timings and the CUDA-event markers are different experiments, neither is request E2E. Six unsupported ctc metrics were reported by NCU; they are unrelated to the collected H200 DRAM/SM/L2 metrics.

## Whole-chain results

| Site | Kernels | GPU time(us) | Speedup | DRAM read+write(bytes) | Traffic reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| norm1 | 6 -> 1 | 338.496 -> 41.024 | 8.251x | 821,483,008 -> 108,945,920 | 86.738% |
| cross_norm | 3 -> 1 | 188.320 -> 40.480 | 4.652x | 475,884,800 -> 91,410,688 | 80.791% |

Both candidates launch 2,340 blocks of 128 threads, two rows per block. Norm1 uses 32 registers/thread with 79.88% active warps,2.12 eligible warps/cycle,56.92% issue-active and 73.06% L2 throughput. Cross norm uses 38 registers/thread with 62.98% active warps,1.55 eligible warps/cycle,52.61% issue-active and 64.58% L2 throughput. Local-memory load/store sectors are zero in both. DRAM active cycles are 55.21% and 46.99% of peak; measured read+write throughput is approximately 2.656 and 2.258 TB/s. The speedup principally comes from avoiding materialized FP32 intermediate tensors and repeated launches, not a faster reduction tree.

## Source and PM evidence

The largest source stall concentration is the epilogue's input reload at line 259: 1,582 norm1 and 976 cross-norm long-scoreboard samples. The statistics pass input conversion/load at line 215 contributes 565/735. The modulation/affine epilogues at lines 290/286 contribute 399/392 long-scoreboard samples associated with their dependencies. These are sampled stall counts, not percentages of execution time. The `_not_issued` metrics are subsets and are retained separately rather than added to their parent counts. PM metrics have separate replay sampling intervals (for example 144/142 long-scoreboard samples); do not compare raw sums as time fractions. Full per-metric values and source maps are retained under analysis/.

This is a memory-traffic reduction with residual load-latency stalls and no spill evidence. A next isolated experiment would keep BF16 input values available for the second normalization pass to avoid the line 259 reload while retaining the exact Welford order. That increases register lifetime across the reduction; the current 32/38 register occupancy and only 1.1/1.48 waves make a regression plausible. It is not included in this patch; no additional speedup is claimed for it.

## Reproduction and scope

Run collect.sh in the recorded environment, then analysis/analyze_ncu.py. The harness, exact metadata, all eight `.ncu-rep` reports, exported details, all metrics, PM samples and source hotspots are included. The CUDA-event marker additionally tests 9,360 tokens and camera/residual sites. Native profile sites include 200 norm1 entries and 199 residual/camera/cross-norm entries across 5 complete forwards. Saved-output CLI and ten/thirty-chunk WebSocket correctness/E2E are reported separately in the PR.
