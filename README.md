# Qwen PR47090: current-head cleanup revalidation incomplete

Current head `188169fa0db0bb2d524d56a97184590f948cca07` passes the frozen 512 gates but fails the frozen 1024 cleanup gate in both T2I and editing. All samples and original thresholds are retained. This report does not qualify the current head. The original PR report remains historical evidence for `efcb88a201a41b1fecc516a60df0ca0a34fb6dd5`.

Frozen SGLang main `af21e2b989e482cfdce39689126562f2209d53ac`; native vLLM-Omni `817d4c17feb702289a2d2b2cbacfd594c262b7a7`; checkpoint `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3`, all 24 runtime files hash verified. Same physical RTX PRO 6000 Blackwell Server GPU0 (GPU-1565b45e-a702-2015-9776-61f68c36fd59), GPU1 idle, driver 595.84. SGLang Torch2.14.1+cu130, Omni Torch2.13.0+cu130. BF16 native components with recorded internal mixed precision, CPU seed1101, 50steps, CFG1, batch/concurrency1, fixed inputs and RGBA PNG semantics.

New primary runs use native resident SGLang eager, SYNC0, without a profiler or observer. Omni retains its native regional compilation and decode graphs. 512 uses native default; 1024 T2I uses the previously quality-qualified native `emulate_precision_casts`, while 1024 editing uses native default. The earlier default1024 T2I quality failure remains in the original PR evidence; it is not removed or counted as a win.

Each resolution has a fresh two-process main A/A followed by the frozen paired primary schedule. Per task: two main and two native Omni processes (16 measured requests each), four original and four current processes (32 each); each process/task also retains four qualification requests, six excluded warmups and eight measured short requests. The second half reverses task order. Total primary: 12 processes/432 PNGs at512 and14 processes/432 PNGs at1024. A/A adds72 PNGs per resolution. Process pairs, not individual requests, are the independent units in paired confidence intervals.

## Primary E2E and memory

| Shape/task | Arm | n | Mean s | p50 s | p95 s | Serial response/s | Peak NVML MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 512 t2i | main | 16 | 2.435254704 | 2.435507814 | 2.438482404 | 0.410635 | 35257 |
| 512 t2i | original | 32 | 2.364894054 | 2.366075612 | 2.372967312 | 0.422852 | 35257 |
| 512 t2i | current | 32 | 2.364191113 | 2.364764111 | 2.371975307 | 0.422978 | 35257 |
| 512 t2i | omni_default | 16 | 2.586238788 | 2.585694485 | 2.591539777 | 0.386662 | 34269 |
| 512 edit | main | 16 | 2.657522689 | 2.656551864 | 2.662917140 | 0.376290 | 35237 |
| 512 edit | original | 32 | 2.578895797 | 2.578749324 | 2.584933220 | 0.387763 | 35237 |
| 512 edit | current | 32 | 2.578449149 | 2.579151311 | 2.584391232 | 0.387830 | 35237 |
| 512 edit | omni_default | 16 | 3.600092034 | 3.599287455 | 3.618790562 | 0.277771 | 38757 |
| 1024 t2i | main | 16 | 9.501570366 | 9.500171912 | 9.514016596 | 0.105246 | 37493 |
| 1024 t2i | original | 32 | 9.402412370 | 9.401856614 | 9.417416885 | 0.106356 | 37517 |
| 1024 t2i | current | 32 | 9.408172547 | 9.410254172 | 9.417993402 | 0.106291 | 37517 |
| 1024 t2i | omni_casts | 16 | 10.516653919 | 10.515226459 | 10.534325720 | 0.095087 | 39843 |
| 1024 edit | main | 16 | 10.977237993 | 10.978408742 | 10.985958670 | 0.091098 | 43541 |
| 1024 edit | original | 32 | 10.873791575 | 10.871960921 | 10.885918304 | 0.091964 | 43541 |
| 1024 edit | current | 32 | 10.878866522 | 10.879124714 | 10.890679547 | 0.091921 | 43541 |
| 1024 edit | omni_default | 16 | 12.363896572 | 12.365365414 | 12.374552425 | 0.080881 | 44149 |

Serial response rate is reciprocal mean serial latency, not a saturation-throughput result. Peak NVML includes process/device residency; it is not tensor allocator size.

## Fixed gates and uncertainty

| Shape/task | E2E reduction vs main | Omni/current mean ratio | Cleanup mean change | Fresh AA noise | Cleanup t95 upper regression | Primary / cleanup |
|---|---:|---:|---:|---:|---:|---|
| 512 t2i | 2.918117% | 1.093921204x | -0.029723959% | 0.163928012% | 0.021939603% | pass / pass |
| 512 edit | 2.975461% | 1.396223786x | -0.017319351% | 0.255535351% | 0.056332464% | pass / pass |
| 1024 t2i | 0.982972% | 1.117821114x | +0.061262760% | 0.017214899% | 0.215976070% | pass / **FAIL** |
| 1024 edit | 0.896140% | 1.136505954x | +0.046671365% | 0.020528947% | 0.095566251% | pass / **FAIL** |

Cleanup requires mean regression no greater than fresh AA noise AND a four-pair log-t95 regression upper bound below0.1%. Both1024 means exceed AA noise; T2I also fails the upper-bound test. The two1024 cleanup intervals include1: this does not prove a nonzero population regression or a source-code cause, and does not satisfy the frozen no-regression assurance. No unchanged primary rerun or relaxed gate was used.

| Shape/task | Ratio | Process pairs | Geomean | Paired log-t95 interval |
|---|---|---:|---:|---|
| 512 t2i | main_over_current | 2 | 1.030205214 | [0.990343236, 1.071671663] |
| 512 t2i | omni_over_current | 2 | 1.094071543 | [1.051070823, 1.138831480] |
| 512 t2i | original_over_current | 4 | 1.000250745 | [0.999780652, 1.000721060] |
| 512 edit | main_over_current | 2 | 1.031055015 | [0.994291071, 1.069178308] |
| 512 edit | omni_over_current | 2 | 1.396808971 | [1.302986223, 1.497387515] |
| 512 edit | original_over_current | 4 | 1.000283287 | [0.999436993, 1.001130298] |
| 1024 t2i | main_over_current | 2 | 1.010063336 | [1.005114707, 1.015036330] |
| 1024 t2i | omni_over_current | 2 | 1.117816311 | [1.113326745, 1.122323981] |
| 1024 t2i | original_over_current | 4 | 0.999415447 | [0.997844894, 1.000988471] |
| 1024 edit | main_over_current | 2 | 1.009509661 | [1.007089446, 1.011935692] |
| 1024 edit | omni_over_current | 2 | 1.137004190 | [1.128434238, 1.145639227] |
| 1024 edit | original_over_current | 4 | 0.999553357 | [0.999045250, 1.000061722] |

Both512 main/current intervals include1 despite their frozen gates passing; only two process pairs support these intervals. Do not claim all main/current intervals exclude1. Both1024 main/current and all four Omni/current intervals exceed1. Independent NumPy/SciPy reviews recomputed all statistics from actual raw results without the admission calculator.

## Startup and excluded warmups

| Shape/task | Arm | Launch-to-ready range s | Sum of six warmup requests per process, range s |
|---|---|---:|---:|
| 512 t2i | main | 44.031–44.044 | 14.635–14.640 |
| 512 t2i | original | 44.029–54.046 | 14.099–14.194 |
| 512 t2i | current | 44.032–52.042 | 14.095–14.188 |
| 512 t2i | omni_default | 34.016–36.022 | 15.517–15.545 |
| 512 edit | main | 44.031–44.044 | 15.963–15.986 |
| 512 edit | original | 44.029–54.046 | 15.403–15.470 |
| 512 edit | current | 44.032–52.042 | 15.401–15.470 |
| 512 edit | omni_default | 34.016–36.022 | 21.569–21.656 |
| 1024 t2i | main | 52.042–52.052 | 56.595–56.742 |
| 1024 t2i | original | 52.042–52.048 | 55.839–56.263 |
| 1024 t2i | current | 52.043–52.047 | 56.113–56.270 |
| 1024 t2i | omni_casts | 40.016–40.018 | 62.842–62.977 |
| 1024 edit | main | 52.042–52.052 | 65.562–65.633 |
| 1024 edit | original | 52.042–52.048 | 64.822–65.070 |
| 1024 edit | current | 52.043–52.047 | 64.887–65.102 |
| 1024 edit | omni_default | 36.017–36.020 | 73.911–74.011 |

Readiness includes native startup/warmup and, for Omni, native compilation/graph setup; these costs are not included in primary saved-request E2E. The same server may handle both tasks; per-task startup entries are not additive.

## Quality

The twelve displayed fresh main/current primary image pairs have identical PNG bytes and RGBA pixels (RGB/alpha SSIM1, PSNR∞). Three distinct prompts per task/shape include short, long and Chinese text; repeated short requests are retained. Pixel equality does not establish semantic perfection. Cross-framework RGB and alpha are assessed separately against0.95 SSIM/28dB thresholds. Below are componentwise minima over the three distinct prompts in the first fresh native Omni process; the full per-prompt records identify potentially different worst samples.

| Shape/task | RGB min SSIM | RGB min PSNR dB | Alpha min SSIM | Alpha min PSNR dB |
|---|---:|---:|---:|---:|
| 512-t2i | 0.991974273 | 38.658691 | 0.999529118 | 63.663978 |
| 512-edit | 0.968473242 | 32.031133 | 0.999083354 | 60.656345 |
| 1024-t2i | 0.972529273 | 29.234914 | 0.999028904 | 60.492056 |
| 1024-edit | 0.997443726 | 51.741815 | 0.999331171 | 62.145452 |

## Separate failure investigation

The complete original→current source diff is two files,16 insertions/12 deletions. Production changes rename the enable flags and pass one outer block flag by keyword instead of position. Arithmetic, kernels and weight storage are unchanged. The actual SM120 packing/restoration/LoRA regression passed at1024 and4096 tokens,2 tests/0 skips. This does not establish full-model LoRA or GPU compile performance.

A separately frozen original/current1024 native high run completed two fresh processes,44 PNGs and four Torch traces, all hash verified. Each task/source retains four qualification requests, two excluded warmups, four SYNC1 diagnostic requests and one separate profiled request. All44 images equal corresponding primary main RGBA outputs. Three complete interior profile steps (#2,#3,#4) per trace have identical GPU kernel sequence/launch geometry, CPU operator sequence/shapes and CUDA runtime counts:384 kernels,32 packed gate/up GEMMs and32 packed SiLU kernels per step,zero graph launches. Original/current sampled peaks are identical within this diagnostic (39339884544B T2I;45669089280B edit). This finds no operator/storage regression mechanism; Python argument-binding overhead is not isolated by these traces.

| Task | Source | SYNC1 denoise p50 s (4 diagnostic requests) | Profile GPU steps2/3/4 ms |
|---|---|---:|---|
| t2i | original | 9.254464985 | 185.254570, 185.646476, 186.028302 |
| t2i | current | 9.228657241 | 185.344291, 185.619332, 185.405699 |
| edit | original | 10.718908526 | 210.659886, 210.604814, 211.152240 |
| edit | current | 10.661360847 | 210.423547, 210.365308, 210.738846 |

These timings are diagnostic, one process per source, and cannot replace primary E2E or prove a performance repair. New resolution-specific main/current stage diagnostics and high→lossless→exact→high/BCG compatibility plans were not run after primary failure. No current-head BCG net benefit, new scenario or complete compatibility validation is claimed. No source patch is justified solely by matching traces.

## Reproduction and archive

Use the pinned commits/checkpoint and an isolated owned GPU0, with the exact native launchers, contracts, warmups, process ordering and pair indices in `evidence/plans/`. `run_qwen21_current_head_blocks1.py` runs each AA or primary plan once; archive/verify/audit AA before primary. Preserve SYNC0 for primary. The separately named `qwen21-cleanup-1024-forensics1` plan uses the unmodified generic diagnostic runner with SYNC1 and explicitly pins the failed primary receipt. Do not treat it as an admission retry. Local root-level remote scripts map to `scripts/` and archived `evidence/reproduction/` paths; remote root is `/data/xyzhang/diffusion-rtx6000-20261026`, mounted `/task`.

Complete archives remain local and are identified by public per-member hashes; public PNGs are deduplicated by content hash, complete raw numerical results/requests/telemetry are retained, response copies omit repeated base64 data with original hashes recorded, and traces are published separately from production code. All historical failures remain in the original PR evidence. The full frozen81-cell matrix is not claimed complete.
