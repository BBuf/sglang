# LongLive 2 Wan normalization post-op NCU evidence

H200 / SM90, CUDA_VISIBLE_DEVICES=0. Shape [1,256,4,240,416], channels-last-3D, BF16 input and gamma, native FP32 norm denominator, FP32 output. This is the dominant production layout in the T2V native trace (415 channels-last fused calls versus 19 NCDHW calls). The denominator is computed before the profiled region in both arms: this report isolates the five replaced post-ops. The registered rotating-buffer benchmark includes the unchanged norm and clamp as well.

Two report sets per arm were collected: full counters with PmSampling/PmSampling_WarpStates and source counters with SourceCounters. The fused harness adds only -lineinfo to the normal --fmad=false build; finite random outputs match integer-bit reference before profiling. Source commit a98b0875c4 is recorded in harness metadata. Final model-source follow-ups reuse the denominator on unsupported fallback and give the custom op a unique name; the CUDA device implementation is unchanged.

| Measurement | Five native post-op kernels | One fused post-op kernel |
| --- | ---: | ---: |
| Sum of kernel duration | 1599.136 us | 315.104 us |
| DRAM reads | 1,842,186,240 B | 206,129,664 B |
| DRAM writes | 1,930,885,632 B | 385,019,392 B |
| Combined DRAM traffic | 3,773,071,872 B | 591,149,056 B |

Fusion cuts isolated post-op time by 80.30% (5.07x) and measured DRAM traffic by 84.33%. The actual memory transactions include cache effects; they are not a tensor-size estimate. The whole-chain rotating benchmark at this shape is 1519.456 -> 364.128 us on channels-last and 1484.960 -> 357.696 us on NCDHW.

The fused launch uses 1056 CTAs of 256 threads, 30 registers/thread, one persistent grid wave, and zero local load/store sectors. Active warps are 70.62% of peak; SM throughput is 65.35% and L2 throughput 47.36%. Combined read/write DRAM throughput is 38.99% of peak, 1.876 TB/s. This is not a saturated-HBM roofline. The persistent loop gives nearly uniform work; no irregular sequence lengths or reduction imbalance are introduced. The kernel contains pointwise scalar math and vector memory operations, with no matrix multiply.

Source counters put the largest primary long-scoreboard counts at denominator consumption (line 42: 7861) and the divide using the loaded input (line 59: 3225). The SiLU line (63) has dependency wait (1946) and MIO-throttle (1749) samples. These are per-PC sampling counters, not percentages or independent wall-time contributions. `stall-hotspots-fused-primary.json` excludes the `_not_issued` subsets and selected/not-selected counters; the initial unfiltered aggregation is retained as raw audit evidence and must not be summed as independent stalls.

PM sampling supplies 334-339 samples per stall series. Long-scoreboard, MIO, and dependency-wait signals dominate the sampled interval; the barrier and LG-throttle series are zero. The series come from separate metric replays and must not be treated as simultaneous lane percentages. Full and source reports plus all scalar metrics and per-instance PM samples are retained under reports/ and analysis/.

One concrete next kernel experiment would specialize the channel count for the channels-last path, replacing runtime 64-bit division/modulo in denominator and gamma indexing with constant arithmetic. It must retain the exact FP32 operations and be checked against the same standalone and native output tests. This experiment is deferred: the present fusion already removes the five large memory passes, while all fused post-ops in the initial complete native T2V profile total only 48.18 ms; subsequent effort is being allocated to the remaining model families after final native validation.
