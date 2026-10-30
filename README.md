# Joy PR47022: current-head square cleanup revalidation failed

This report concerns official head b08c24e9630c63790c9f86b48d6c7fc10a0c4301. It preserves the failed frozen cleanup gate. It does not qualify this head, replace the original tested-head report, claim full shape/quality coverage, or add an optimization PR.

Frozen SGLang main: af21e2b989e482cfdce39689126562f2209d53ac. Original tested head: 4c135ea92d6b4cbb0ead69bb39976f7dbe995455. Native vLLM-Omni: 817d4c17feb702289a2d2b2cbacfd594c262b7a7, regional compile and CUDNN_ATTN retained.

Same physical RTX PRO 6000 Blackwell Server GPU0 (GPU-1565b45e-a702-2015-9776-61f68c36fd59), GPU1 idle; BF16 DiT/encoder/VAE, native backend, high quality, 1024x1024, 30 steps, CFG4, seed1101, batch/concurrency1. Checkpoint 4b41fb25d961f37668750178ccbb380da326201c. Four qualification requests, two excluded warmups and four measured short requests per process. All fresh SG primary arms use SYNC0 without observer/profiler; historical SYNC1 numbers are not pooled.

Fixed primary process order: main-original-current-Omni-original-current-current-original-Omni-current-original-main. Main/native Omni have two process replicates each; original/current have four each. All 12 processes and 120 output PNGs completed and remain in the verified raw archive. Separate preceding AA has two main processes and 20 PNGs.

| Arm | Measured requests | Mean E2E (s) | p50 (s) | p95 (s) | Peak NVML MiB |
|---|---:|---:|---:|---:|---:|
| main | 8 | 35.983319873 | 35.982784805 | 35.987944768 | 60223 |
| candidate_original | 16 | 35.302859694 | 35.303303601 | 35.315088409 | 59805 |
| candidate_current | 16 | 35.319337247 | 35.312821466 | 35.343363638 | 59943 |
| omni_compile | 8 | 37.719426879 | 37.719557579 | 37.725595533 | 66735 |

Current remains 1.845251% lower-latency than frozen main and native Omni/current is 1.067953983x. Both main/current and Omni/current pairs improve. These positive comparator results do not satisfy the separate cleanup gate.

Cleanup mean regression: 0.046674838% vs fresh AA noise 0.008928383%. Four-process-pair log-t95 upper regression: 0.110357171%, above the frozen <0.1% limit. Both cleanup checks fail. Every sample and original threshold is retained.

Every current-head process samples a peak 138 MiB above the original tested head. A separate unprofiled actual SM120 production-MLP probe confirms current retains its temporary normalized input through fc_out, while original releases it before fc_out: allocator differences are exactly 56/64 MiB for 7168/8192 BF16 tokens. All eight probe outputs agree, and eight separate traces have matching two-kernel dispatches per shape. This proves input-lifetime memory impact; it does not attribute the entire 138 MiB NVML difference or measured E2E difference to that one cause.

Actual current/main high-quality square outputs pass the frozen numerical image checks. The three displayed pairs come from fresh primary qualification requests, with complete 1:1 image pixels and source identities in media/manifest.json. Fresh wide/tall, quality-switching and synchronized denoise/profile supplement for this official head were not run after this failure; no new BCG or full-model LoRA/compile result is claimed. A separately identified minimal repair is being tested and is not evidence that official b08c passed.

Reproduction: install the recorded source commits and original checkpoint, preserve inputs and native launchers, then run run_joy_current_head_blocks1.py with joy-current-head-square-aa1.json. Archive/verify/analyze/admit AA before the primary plan. Use a fresh output directory, retaining all fixed process pairs and requests; do not overwrite these runs. SYNC0 stage perf dumps are retained but do not establish denoise-stage latency.

Complete raw archives remain local with public per-member SHA256 manifests. Raw numerical records, source/ownership metadata, telemetry, launchers, plans, admissions, relevant source diff and independent lifetime diagnostic are exported. Unselected full PNGs and repeated response JSON are omitted here with original hashes recorded; representative actual source PNGs remain included. No checkpoint weights or transfer bundles are published. All artifacts concern failed/partial validation, not a successful final qualification.
