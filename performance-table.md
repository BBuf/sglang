| Configuration | Prompt | n | Mean / p50 / p95 E2E (s) | Serial req/s | Request peak GiB | Separate synchronized denoise / decode (s) |
|---|---|---:|---:|---:|---:|---:|
| omni_eager | short | 6 | 85.2435 / 85.1708 / 85.4637 | 0.01173 | 31.011 | 81.1538 / 3.2919 (n=1) |
| omni_eager | long | 6 | 85.2417 / 85.1557 / 85.4571 | 0.01173 | 31.011 | 81.1240 / 3.2956 (n=1) |
| main_eager | short | 6 | 63.5638 / 63.5646 / 63.5718 | 0.01573 | 23.054 | 58.4761 / 5.0376 (n=4) |
| main_eager | long | 6 | 63.5655 / 63.5674 / 63.5722 | 0.01573 | 23.054 | 58.4810 / 5.0345 (n=2) |
| v1_eager | short | 12 | 61.3030 / 61.3000 / 61.3237 | 0.01631 | 23.025 | not collected here |
| v1_eager | long | 12 | 61.3000 / 61.2966 / 61.3201 | 0.01631 | 23.025 | not collected here |
| final_eager | short | 12 | 61.3031 / 61.3028 / 61.3103 | 0.01631 | 23.025 | 56.1882 / 5.0333 (n=4) |
| final_eager | long | 12 | 61.3054 / 61.3074 / 61.3156 | 0.01631 | 23.025 | 56.1962 / 5.0347 (n=2) |

Primary values use every predeclared measured request. NVML is sampled every 250 ms during requests; these are observed device-memory peaks, not exact allocator maxima. Denoise is from a separate synchronized diagnostic, not a component subtracted from primary E2E. V1 is retained for cleanup comparison. Omni denoise/decode reuses the already archived native stage diagnostic at the unchanged frozen source/checkpoint/configuration, with one steady request per English prompt. All primary E2E samples are new. Native stage boundaries differ across frameworks; no diagnostic timing replaces primary E2E.

| Prompt | Final mean E2E reduction vs main | Omni / final mean E2E | V1 → final mean latency change | Cleanup paired-t 95% regression upper bound |
|---|---:|---:|---:|---:|
| short | 3.55649% | 1.390525× | +0.00028% | 0.02661% |
| long | 3.55557% | 1.390443× | +0.00892% | 0.03643% |

| Comparison | Prompt | Process pairs | Paired median geometric ratio | Paired-log-t 95% interval |
|---|---|---:|---:|---:|
| main_vs_final | short | 2 | 1.036879 | [1.036786, 1.036973] |
| omni_vs_final | short | 2 | 1.391066 | [1.355012, 1.428079] |
| omni_vs_main | short | 2 | 1.341590 | [1.306936, 1.377162] |
| v1_vs_final_cleanup | short | 4 | 0.999996 | [0.999734, 1.000258] |
| main_vs_final | long | 2 | 1.036920 | [1.035395, 1.038448] |
| omni_vs_final | long | 2 | 1.391161 | [1.354252, 1.429074] |
| omni_vs_main | long | 2 | 1.341627 | [1.307958, 1.376164] |
| v1_vs_final_cleanup | long | 4 | 0.999892 | [0.999636, 1.000149] |

Intervals use process-pair log ratios, not requests as independent process replicates; only two main/final and Omni/final process pairs limit precision. Cleanup has four predeclared pairs. All original requests and warmups remain in raw evidence.

| Configuration | Cached launch-to-ready min / max (s) |
|---|---:|
| main_eager | 104.126 / 104.140 |
| v1_eager | 100.116 / 102.132 |
| final_eager | 100.105 / 102.126 |
| omni_eager | 26.014 / 28.016 |

These are cached process launches, not cold-start distributions. BCG capture/replay costs and quality transitions are reported separately; this depthwise fusion does not establish new BCG support or a BCG net win.
