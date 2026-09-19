# SenseNova-U1.5 lossless RoPE on H200

Native eager, 2048x2048, 50 steps, CFG 4, seed 42, bf16, one H200. Baseline e8d68ce11f (993d1fccba plus profiling controls); candidate 853522eff1.

Second isolated ABBA: worker request E2E 20.407937 -> 19.353700s (5.17% reduction); client request including output save 20.870 -> 19.800s (5.13%).

[Complete measurements and numerical audit](final-evidence.json), [operator benchmark](rope-committed-microbench.json), [environment](environment.json), [weight cleanup](cache-cleanup.json). The first ABBA is retained in raw/ and final-evidence.json; its first client request had higher non-worker overhead. Profile wall time is diagnostic and excluded from speed claims.

| Before | After |
| --- | --- |
| ![Before](baseline.png) | ![After](candidate.png) |

All eight unprofiled and two profiled PNG files and decoded pixels are identical. The native pipeline disables BCG; no fallback timing is accepted.

The two compact Chrome traces contain a complete denoising cycle with zero crossing GPU kernels. Full raw traces are retained in the campaign archive; their SHA256 manifests identify the originals. Per-slice evidence JSON records the extraction rule, exact boundaries and source-attributed kernel totals. The automated triage suggestion associating nvjet with FP8 is a known false positive for this BF16 model; no GEMM change is made.

The lossless application fusion reduces source-attributed RoPE kernels from 2520 to 840 and cumulative GPU time from 28.999471 to 5.667296ms per cycle, including 336 layout copies. No kernel arithmetic was changed.
