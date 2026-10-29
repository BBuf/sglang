| Configuration | Prompt | n | Mean / p50 / p95 E2E (s) | Serial req/s | Request peak GiB | Separate synchronized denoise / decode (s) |
|---|---|---:|---:|---:|---:|---:|
| omni_eager | short | 6 | 85.1189 / 85.1176 / 85.1365 | 0.01175 | 31.011 | 81.1538 / 3.2919 (n=1) |
| omni_eager | long | 6 | 85.1154 / 85.1145 / 85.1263 | 0.01175 | 31.011 | 81.1240 / 3.2956 (n=1) |
| main_eager | short | 6 | 63.5963 / 63.5936 / 63.6156 | 0.01572 | 23.054 | 58.5181 / 5.0380 (n=4) |
| main_eager | long | 6 | 63.6024 / 63.6025 / 63.6270 | 0.01572 | 23.054 | 58.5239 / 5.0381 (n=2) |
| v1_eager | short | 12 | 61.8100 / 61.8112 / 61.8227 | 0.01618 | 23.054 | not collected here |
| v1_eager | long | 12 | 61.8118 / 61.8131 / 61.8330 | 0.01618 | 23.054 | not collected here |
| final_eager | short | 12 | 61.8130 / 61.8119 / 61.8314 | 0.01618 | 23.054 | 56.7082 / 5.0372 (n=4) |
| final_eager | long | 12 | 61.8121 / 61.8121 / 61.8288 | 0.01618 | 23.054 | 56.7166 / 5.0373 (n=2) |

Primary values use every predeclared measured request. NVML is sampled every 250 ms during requests; these are observed device-memory peaks, not exact allocator maxima. Denoise is from a separate synchronized diagnostic, not a component subtracted from primary E2E. V1 is retained for cleanup comparison. Omni uses its native synchronized method wrappers, with one steady diagnostic request per English prompt. Native stage boundaries differ across frameworks; no diagnostic timing replaces primary E2E.

| Prompt | Final mean E2E reduction vs main | Omni / final mean E2E | V1 → final mean latency change | Cleanup paired-t 95% regression upper bound |
|---|---:|---:|---:|---:|
| short | 2.80412% | 1.377038× | +0.00492% | 0.02248% |
| long | 2.81471% | 1.377001× | +0.00059% | 0.04267% |

| Comparison | Prompt | Process pairs | Paired median geometric ratio | Paired-log-t 95% interval |
|---|---|---:|---:|---:|
| main_vs_final | short | 2 | 1.028797 | [1.028197, 1.029398] |
| omni_vs_final | short | 2 | 1.376901 | [1.374693, 1.379112] |
| omni_vs_main | short | 2 | 1.338359 | [1.336993, 1.339727] |
| v1_vs_final_cleanup | short | 4 | 0.999899 | [0.999775, 1.000023] |
| main_vs_final | long | 2 | 1.028919 | [1.026809, 1.031033] |
| omni_vs_final | long | 2 | 1.376851 | [1.372864, 1.380849] |
| omni_vs_main | long | 2 | 1.338153 | [1.331543, 1.344796] |
| v1_vs_final_cleanup | long | 4 | 0.999953 | [0.999573, 1.000333] |

Intervals use process-pair log ratios, not requests as independent process replicates; only two main/final and Omni/final process pairs limit precision. Cleanup has four predeclared pairs. All original requests and warmups remain in raw evidence.

| Configuration | Cached launch-to-ready min / max (s) |
|---|---:|
| main_eager | 104.121 / 104.133 |
| v1_eager | 102.120 / 102.126 |
| final_eager | 102.119 / 102.126 |
| omni_eager | 28.014 / 28.015 |

These are cached process launches, not cold-start distributions. BCG capture/replay costs and quality transitions are reported separately; this eager opt-in is not new BCG support.
