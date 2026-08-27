# PR #36680: two-H200 all-model evidence

This orphan branch contains generated-media contact sheets, structured ABBA
summaries, and cleanup ledgers for the cross-model audit of
[`sgl-project/sglang#36680`](https://github.com/sgl-project/sglang/pull/36680).
It is deliberately separate from the PR branch.

## Fixed comparison

- Base: `a7e3f590cabb676a02181fb3328026587822f934`
- PR: `8e056cfbfe68ddc46c1f869b7853ec9fc7e94cfe`
- Hardware: the same physical NVIDIA H200 pair, GPUs 6/7
- Order: base-a, PR-a, PR-b, base-b; reported values are two-run medians
- Native SGLang backend, lossless output, fixed prompt/seed and checked-in
  benchmark preset; eager with compile/BCG off unless the row says BCG
- Each model used a task-owned cache. It was deleted and verified empty before
  the next model; see both cleanup ledgers.

## Affected, successfully executed checkpoints

| Preset / checkpoint | Topology | Base denoise (s) | PR denoise (s) | Change | Output result |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen-Image-2512 (`qwen`) | TP2 + BCG | 8.5406 | 7.8657 | **+8.58%** | repeats stable; Base/PR SSIM 0.9713 |
| Qwen-Image (`qwen-image`) | 1 GPU + BCG | 12.6528 | 12.3580 | **+2.39%** | bit-exact |
| Qwen-Image-Edit | 1 GPU eager | 32.0573 | 32.2576 | -0.62% | bit-exact |
| Qwen-Image-Edit-2509 | 1 GPU eager | 22.3352 | 22.3464 | -0.05% | bit-exact |
| Qwen-Image-Edit-2511 | TP2 eager | 14.2704 | 14.2651 | +0.04% | bit-exact |
| Qwen-Image-Layered | 1 GPU eager | 32.9077 | 32.8971 | +0.03% | repeats stable; min SSIM 0.9891 over 4 layers |
| FireRed-Image-Edit-1.0 | CFG2 eager | 11.4178 | 11.4694 | -0.45% | bit-exact |
| FireRed-Image-Edit-1.1 | CFG2 eager | 11.4151 | 11.4170 | -0.02% | bit-exact |
| Z-Image-Turbo | TP2 eager | 0.5575 | 0.5307 | **+5.05%** | bit-exact |
| Z-Image | 1 GPU eager | 9.2629 | 9.3055 | -0.46% | bit-exact |
| Cosmos3-Super T2V | TP2 eager | 113.2269 | 113.0971 | +0.11% | MP4 and sampled frames bit-exact |
| Cosmos3-Super I2V | TP2 eager | 112.9026 | 112.9645 | -0.05% | MP4 and sampled frames bit-exact |
| ERNIE-Image | 1 GPU eager | 13.0868 | 12.8640 | **+1.73%** | prompt enhancement nondeterministic on both sides |
| ERNIE-Image-Turbo | 1 GPU eager | 12.9039 | 13.0717 | -1.28% | prompt enhancement nondeterministic on both sides |
| LingBot Video MoE 30B | 1 GPU eager | 3.9435 | 3.9126 | +0.79% | MP4 and sampled frames bit-exact |

Positive percentages mean lower PR latency. The bold rows cross the audit's
1.5% signal threshold. In particular, the complete Qwen checkpoint sweep does
**not** show a family-wide speedup: the clear wins are the two Qwen T2I BCG
lanes, while edit/layered/FireRed lanes are neutral or slightly slower.

## Access- or hardware-blocked affected presets

| Preset | Why no model-backed number |
| --- | --- |
| FLUX.1-dev TP2 | Hugging Face 403 gated repository |
| FLUX.2-dev TP2 | Hugging Face 403 gated repository |
| Ideogram 4 FP8 TP2 | Hugging Face 403 gated repository |
| Ideogram 4 Fast / Instant | public wrapper resolves to gated Ideogram weights (403) |
| Krea-2 registry root | Registry placeholder returns Hugging Face 404; no checkpoint exists at that ID |
| Krea-2 Turbo / Raw | Hugging Face 403 gated repositories |
| MiniMax-H3 TP2+Ulysses2 | checked-in preset needs 4 GPUs; assignment had 2 |
| Cosmos3-Super T2V CFG2+TP2 | preset needs 4 GPUs; assignment had 2 |
| Cosmos3-Super distilled T2I TP4 | preset needs 4 GPUs; assignment had 2 |
| Qwen-Image NVFP4 / Ideogram 4 NVFP4 | Blackwell-only quantized paths; H200 cannot execute them |

Every 403 attempt also used an isolated cache and left zero bytes behind.

## Exploratory negative controls (not attributed to the PR)

| Preset | Base (s) | PR (s) | Observed change | Why it is a control |
| --- | ---: | ---: | ---: | --- |
| FLUX.2 Klein 4B | 0.2390 | 0.2332 | +2.52% | 1 GPU, no masked-varlen metadata |
| FLUX.2 Klein Base 4B | 6.0333 | 5.9889 | +0.74% | 1 GPU, no masked-varlen metadata |
| LTX-2 | 9.0477 | 9.4805 | -4.57% | CFG2, not TP; no varlen metadata |
| LTX-2.3 two-stage | 13.5873 | 12.7691 | +6.41% | CFG2, not TP; no varlen metadata |

These controls show why small or isolated ABBA shifts must not be generalized
without proving that the changed branch executed.

## Files

- `images/`: Base/PR contact sheets; video sheets use first and middle frames.
- `results/`: per-run medians, commits, topology, output hashes, and comparison
  metrics for every successful model.
- `cleanup-ledger-early.jsonl` and `cleanup-ledger-late.jsonl`: 29 cleanup
  records; every record reports zero post-cleanup bytes and weight files.
