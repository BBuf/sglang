# Joy Image Edit: joint image/text QKV copy on H200

This profile isolates a pure-copy optimization after the existing Q/K normalization and RoPE calls. Native input shapes are image `[1, 8048, 32, 128]` and text `[1, 1004, 32, 128]`, BF16. Q/K rows are contiguous; V rows retain the packed projection stride of 12288 elements. Outputs retain image-first order. Baseline is SGLang `80da4432d085ed4d6166ef643d9fd2b829dbb0c5`; the measured implementation is `0cff694f701987ba694954a4cf54257f7c3881e6`. The final commit `3b00154ddb33ac436906a493c9706ed3855ec918` changes lazy-import portability guards and documentation, with separate end-to-end and correctness validation.

## 1. Workload and bottleneck

The baseline native model trace has 120 concatenation kernels per 40-block forward, totaling 13.079652 ms. Its 40 generic strided V cats account for 10.104390 ms; the 80 Q/K cats account for 2.975262 ms. The candidate trace has 40 joint copy kernels totaling 4.366372 ms. These are summed CUDA kernel durations within one complete forward, not end-to-end latency. No arithmetic or logical tensor-copy volume is removed.

## 2. Timing and memory throughput

| Isolated NCU workload | Kernels | Duration | DRAM read bytes | DRAM write bytes |
|---|---:|---:|---:|---:|
| Native three cats | 3 | 401.472 us | 222,510,592 | 161,069,568 |
| Joint QKV copy | 1 | 108.832 us | 222,477,312 | 199,803,648 |

The native Q/K cats take 36.064 and 36.448 us; V takes 328.960 us. The generic V cat reaches 53.04% SM throughput but only 12.10% L2 throughput; the candidate reaches 10.82% SM and 88.90% L2 throughput. This supports the inference that a simpler row-wise copy removes substantial generic strided indexing/instruction overhead. It does not show lower DRAM traffic: measured write bytes actually increase, and cache/replay conditions affect these counters. NCU replay timing is separate from native request timing.

The unprofiled, normally allocating CUDA-event microbenchmark (10 warmups, median of 40) is 292.016 to 174.768 us for the production shape. Standalone CUDA graph timing is 285.664 to 114.416 us; this is a kernel diagnostic, not native model BCG. At the 32 MiB model dispatch boundary, eager timing is 156.000 to 120.960 us. Smaller inputs use the native path: 87.792 to 88.112 us and 31.344 to 31.648 us. The earlier unguarded version regressed those small eager cases and is retained in the evidence.

## 3. Launch geometry and resource pressure

The candidate launches 27,156 CTAs of 128 threads, one per output token and Q/K/V component, with 34 registers per thread and 17.14 waves per SM. Native V uses 528 CTAs of 512 threads, 25 registers, and one wave per SM. Local-memory load/store sectors are zero for all measured kernels: there is no measured register spilling. The candidate's achieved active-warps metric is 59.72%, below native V's 71.64%; higher occupancy alone would therefore mispredict this result.

## 4. Scheduler and stall evidence

Native V has 3.166 eligible warps per active cycle and 67.43% issue activity; the candidate has 0.164 and 11.28%. Candidate long-scoreboard and drain ratios are 49.92 and 11.58 per issued instruction. These values are consistent with a shorter memory-copy kernel waiting on memory rather than doing the generic kernel's indexing work. They do not justify maximizing arithmetic issue rate or occupancy at the expense of elapsed time.

## 5. Source correlation

The source-counter report attributes 4,368 samples to the image load (line 47), including 3,717 long-scoreboard samples. The output store (line 55) has 3,806 samples, including 1,600 drain and 1,489 long-scoreboard samples. The text load has 575 samples, 508 long-scoreboard. The Q/K/V branch has 812 samples, mainly short-scoreboard. No numerical transformation is introduced; integer-view correctness tests also cover signed zero, subnormals, infinities and NaN payloads.

## 6. PM sampling and interpretation limits

The full report includes PM sampling. The helper's default metric keys found no instances; this failed extraction is retained in `analysis/pm-default-keys.txt`. The actual report exposes `LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed` and `FBSP.TriageCompute.dramc__throughput.avg.pct_of_peak_sustained_elapsed`. Extraction with those names succeeds (215 instances, 214 active L2 samples and 211 active DRAM samples). The ASCII plot includes pre/post-kernel sampling windows and its bucketing does not display every sample. It cannot support a precise tail-balance claim without timestamp alignment; use the report in Nsight Compute for that analysis. The six unavailable Hopper CTC metrics reported during collection are not used above.

## Decision

Keep the large-tensor guard and verify the final commit with repeated saved-output native requests. The first single request improved worker latency by only about 1.1%; it remains non-qualifying evidence. The two full-warm v2 groups exceed 1.5% worker/client improvement, but the first group's client baseline contains a 15.36 s outlier, so it must not be used to advertise a 4.6% client gain. Native Joy BCG reports disabled and is excluded. Further kernel tuning would target memory copy layout or launch overhead; neither lowering occupancy nor changing arithmetic is justified by these measurements. No additional edit is needed before the final independent end-to-end result is available.

Reproduction files: `harness/bench-joy-qkv-cat-v2.py`, `reports/baseline.ncu-rep`, `reports/candidate.ncu-rep`, `reports/candidate-source.ncu-rep`, and all raw/extracted metrics under `analysis/`.
