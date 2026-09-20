# Joy Image Edit: strided image QK/RoPE and joint QKV concatenation

Revision `4811f4d52aa2586412f699b3bb84ed184d76250a`, baseline `80da4432d085ed4d6166ef643d9fd2b829dbb0c5`, NVIDIA H200. The measured chain consists of the image Q/K contiguous copies, existing QK-Norm/interleaved RoPE, and the three image/text concatenations. Image input is BF16 `[1,8048,32,128]`, packed token stride 12288; normalized text Q/K and packed V are `[1,1004,32,128]`. No text normalization, attention or GEMM timing is included in this focused harness.

## 1. Profile-driven change

The native baseline has 80 image Q/K copy kernels, 40 image QK-Norm/RoPE kernels, and 120 Q/K/V concatenations per complete 40-block forward. The copy-only prototype removed 80 concatenation launches but its final first ABBA group did not meet the fixed 1.5% worker/client threshold. This revision also removes the two preliminary image Q/K copies per block by reusing the existing out-of-place CUDA operation. It retains that operation's arithmetic and contiguous outputs; inputs remain pristine for safe fallback.

## 2. Elapsed time and memory traffic

| NCU chain | Kernels | GPU duration | DRAM read | DRAM write |
|---|---:|---:|---:|---:|
| Original copies + norm/RoPE + three cats | 7 | 678.560 us | 490,464,000 B | 364,624,640 B |
| Out-of-place norm/RoPE + joint copy | 3 | 229.568 us | 358,565,632 B | 312,448,256 B |

Measured DRAM read+write falls 21.53%, from 855,088,640 to 671,013,888 bytes. Unlike the earlier copy-only optimization, the combined chain actually removes two input copy passes. These are replay/counter measurements; do not substitute them for native request E2E.

The two native input copies take 84.256 and 84.864 us. Native image QK-Norm/RoPE takes 108.512 us; its out-of-place variant takes 118.368 us. The variant itself is slightly slower; omitting the copies more than compensates. Original Q/K cats take 35.840 and 35.488 us, while the generic strided V cat takes 327.392 us. One joint copy takes 109.152 us. Position arange remains in both arms (2.208 versus 2.048 us).

Unprofiled production-helper microbenchmark, normally allocating outputs, median of 40 CUDA-event samples after 10 warmups:

| B / image tokens / text tokens | Eager native → candidate | Standalone graph native → candidate |
|---|---:|---:|
| 1 / 8048 / 1004 | 541.360 → 271.440 us | 531.408 → 213.648 us |
| 1 / 4096 / 512 | 288.880 → 190.192 us | 278.688 → 112.912 us |
| 1 / 2048 / 256 | 154.416 → 157.264 us | 145.312 → 145.376 us |
| 2 / 257 / 13 | 110.880 → 118.432 us | 46.656 → 46.768 us |

Both smaller shapes retain the native operations. Their eager rows expose the extra wrapper/guard overhead (2.848 and 7.552 us); no small-shape speedup or exact performance neutrality is claimed. The standalone graph rows are kernel diagnostics, not model BCG support.

## 3. Launch geometry and resources

Both norm/RoPE variants use 1,056 CTAs, 256 threads, 32 registers/thread and one wave per SM. Active warps are 94.07% native and 89.47% out-of-place. The new joint-copy kernel uses 27,156 CTAs, 128 threads, 34 registers/thread and 17.14 waves per SM; active warps are 59.94%. All measured local-memory load/store sector counts are zero, providing no evidence of spills.

The original generic V cat uses 528 CTAs of 512 threads, 25 registers/thread and one wave per SM. It reaches 12.18% L2 throughput, whereas the joint copy reaches 88.94%. This supports the inference that the specialized row copy spends less work on generic indexing and makes better use of the memory subsystem.

## 4. Scheduler evidence

Native norm/RoPE has 2.338 eligible warps/active cycle and 70.02% issue activity; out-of-place has 2.569 and 69.55%. The joint copy has 0.167 eligible warps and 11.44% issue activity. Its much lower issue activity is compatible with a faster memory-copy workload; maximizing issue rate would be the wrong objective. The copy-only prototype's initial small-input regressions also show why a faster GPU kernel does not guarantee a faster eager helper.

## 5. Source and stall correlation

The source report contains three actions. The generic extraction helper defaults to action zero (position arange), so `extract-joy-outplace-source-pm.py` explicitly selects action 1 for norm/RoPE and action 2 for joint copy. The joint-copy source has 12 mapped lines: the image load at line 47 has 4,301 samples, including 3,659 long-scoreboard; the store at line 55 has 3,797, including 1,639 drain and 1,468 long-scoreboard; the text load has 578, including 510 long-scoreboard.

The existing CUDA JIT norm/RoPE build has no usable source-line mapping in this report. Its 10,757 sampled stalls aggregate at unknown line 0 (6,129 long-scoreboard). This is a source-mapping limitation, not evidence locating a specific CUDA statement. The shared CUDA template was reviewed directly to verify that the out-of-place variant changes output addressing while preserving the normalization and rotary arithmetic. No new arithmetic edit is justified by the unmapped stall sample.

## 6. PM sampling and limits

PM sampling is present for all focused kernels. The report uses `LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed` and `FBSP.TriageCompute.dramc__throughput.avg.pct_of_peak_sustained_elapsed`. Raw arrays are preserved in `analysis/pm-instances.json`; sample counts are 202 for baseline norm/RoPE, 371 for baseline V cat, 213 for candidate norm/RoPE and 208 for candidate joint copy. The ASCII exporter uses one column per sample to avoid truncating a remainder. Sampling windows still include pre/post-kernel periods; no precise tail-balance conclusion is drawn without timestamp alignment.

## Decision and acceptance

The combined chain gives a materially larger microbenchmark margin than copy-only. The final native profile and exact saved-image checks pass; actual model BCG remains disabled. Default worker-save oneshot and persistent client comparisons failed and are retained, including all request-tail outliers. A separate diagnostic baseline shows a 1.76 s output-rank empty_cache delay; it does not prove the cause of every earlier candidate outlier.

Final uninstrumented qualification uses the existing native client-save mode in both revisions (return_file_paths_only=False, save_output=True). Two predeclared ABBA groups, each twenty measured requests plus four saved warmups, reduce worker E2E by 3.16% / 3.27% and client E2E including actual PNG saving by 3.13% / 3.25%; unrounded outer wall also improves by 3.12% / 3.25%. This supports publication for that output configuration, with no default-path saved-client speedup claim. Keep the narrow hardware/layout guards; no further arithmetic or launch edit is supported by these counters.

Raw reports: `reports/baseline.ncu-rep`, `reports/candidate.ncu-rep`, `reports/candidate-source.ncu-rep`. The harness is `harness/bench-joy-outplace-v1.py`. All per-kernel metrics and source/PM exports are under `analysis/`.
