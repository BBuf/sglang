Candidate current in these tables is the isolated repair03368cf1c325e526184b8fb223443e1ae931b13c, not official merged PR47022 headb08c24e9630c63790c9f86b48d6c7fc10a0c4301. The official-head square cleanup failure remains recorded; no qualification count or new optimization slot is assigned.

All primary values below are new SYNC0 requests. Historical Joy timings used SYNC1 and are not pooled with these measurements.

| Shape | Configuration | n | Mean / p50 / p95 HTTP E2E (s) | Serial req/s | Observed request peak GiB |
|---|---|---:|---:|---:|---:|
| square | main | 8 | 35.9856 / 35.9857 / 35.9942 | 0.02779 | 58.812 |
| square | candidate_original | 16 | 35.3102 / 35.3102 / 35.3320 | 0.02832 | 58.404 |
| square | candidate_current | 16 | 35.3017 / 35.3021 / 35.3133 | 0.02833 | 58.404 |
| square | omni_compile | 8 | 37.7186 / 37.7202 / 37.7250 | 0.02651 | 65.171 |
| wide | main | 8 | 30.4595 / 30.4620 / 30.4708 | 0.03283 | 57.371 |
| wide | candidate_original | 16 | 29.8475 / 29.8489 / 29.8582 | 0.03350 | 57.335 |
| wide | candidate_current | 16 | 29.8502 / 29.8492 / 29.8605 | 0.03350 | 57.335 |
| wide | omni_compile | 8 | 31.8081 / 31.8044 / 31.8182 | 0.03144 | 63.990 |

| Shape | A/A noise | Repair E2E reduction vs main | Omni / repair | Original → repair mean latency change | Cleanup paired-t95 upper regression |
|---|---:|---:|---:|---:|---:|
| square | 0.005939% | 1.900384% | 1.068462× | -0.023962% | 0.001619% |
| wide | 0.026688% | 2.000340% | 1.065589× | +0.009131% | 0.023574% |

| Shape | Comparison | Process pairs | Paired median geometric ratio | Log-t95 interval |
|---|---|---:|---:|---:|
| square | main_over_current | 2 | 1.019217 | [1.018615, 1.019818] |
| square | omni_over_current | 2 | 1.068330 | [1.067606, 1.069054] |
| square | original_over_current | 4 | 1.000312 | [0.999984, 1.000641] |
| wide | main_over_current | 2 | 1.020458 | [1.019957, 1.020960] |
| wide | omni_over_current | 2 | 1.065591 | [1.064649, 1.066533] |
| wide | original_over_current | 4 | 0.999948 | [0.999764, 1.000132] |

Every frozen request and process pair is retained. Main and native Omni have two process replicates per shape; original/current have four. Intervals use process pairs, not individual requests as independent replicates. NVML peaks are sampled observations, not allocator maxima. Serial response rate is 1/mean E2E at concurrency1.

| Shape | Source | Separate synchronized short denoise p50 (s) | Stage samples | Cached launch-to-ready (s) |
|---|---|---:|---:|---:|
| square | main | 35.6439 | 2 | 80.100 |
| square | current | 34.9595 | 2 | 80.093 |
| wide | main | 30.1533 | 2 | 80.096 |
| wide | current | 29.5308 | 2 | 80.092 |
| tall | main | 30.1501 | 2 | 80.086 |
| tall | current | 29.5319 | 2 | 80.091 |

These SYNC1 stage values come from separate lossless→high→lossless diagnostic processes and never replace primary E2E. Tall is a quality/shape check only. Previously published native Omni stage diagnostics remain historical; no new Omni stage measurement is claimed.

| Profile | Kernels | Kernel sum (ms) | ATen GELU calls | Fused custom-op calls | Fused activation kernels |
|---|---:|---:|---:|---:|---:|
| square-main | 2306 | 1187.550 | 162 | 0 | 0 |
| square-current | 2226 | 1164.016 | 82 | 80 | 80 |
| wide-main | 2306 | 1004.334 | 162 | 0 | 0 |
| wide-current | 2226 | 983.052 | 82 | 80 | 80 |

| Shape | Quality | Samples across phases | Pixel-exact samples | Minimum SSIM | Minimum finite PSNR (dB) |
|---|---|---:|---:|---:|---:|
| square | high | 4 | 0 | 0.998134 | 55.23274 |
| square | lossless | 8 | 8 | 1.000000 | all infinite |
| wide | high | 4 | 0 | 0.998059 | 55.67346 |
| wide | lossless | 8 | 8 | 1.000000 | all infinite |
| tall | high | 4 | 0 | 0.998272 | 56.10906 |
| tall | lossless | 8 | 8 | 1.000000 | all infinite |

All native outputs reproduce the pinned same-source references. Lossless compares exact RGB pixels; high checks every request against SSIM≥0.95 and PSNR≥28dB. Short/long/Chinese/repeated-short requests and return to lossless are retained. These data do not establish new BCG, GPU compile performance or full-model LoRA support. Actual main/repair media review and a publication decision remain separate gates.
