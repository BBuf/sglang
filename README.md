# Joy Image Edit actual main / final candidate outputs

These are whole, unmodified generated images placed side by side at1:1 RGB pixels, with labels outside their bounds. They are not illustrations. All three comparisons were visually inspected locally.

- Model: `jdopensource/JoyAI-Image-Edit-Diffusers@4b41fb25d961f37668750178ccbb380da326201c`.
- SGLang main: `af21e2b989e482cfdce39689126562f2209d53ac`.
- Candidate: `4c135ea92d6b4cbb0ead69bb39976f7dbe995455`.
- RTX PRO6000 Blackwell Server,1024x1024,30steps,CFG4,CPU seed1101,1GPU,batch1,concurrency1. BF16 DiT/encoder/VAE, native mixed precision, high quality.
- `joy-workloads.json` records exact prompts and the reference-image hash; `media-manifest.json` ties every panel to original PNG hashes, source heads and contract hash.
- Candidate outputs come from the actual committed-head cleanup retest. Baseline outputs come from the pinned main qualification.

![Short edit](00-edit-short-main-vs-pr.png)
![Long edit](01-edit-long-main-vs-pr.png)
![Chinese edit](02-edit-chinese-main-vs-pr.png)

Shared baseline appearance limitations are retained; no rerolls, crops, recoloring or illustrative substitutions. Main/candidate high-mode outputs are not bitwise identical. Per-image SSIM/PSNR are printed on each panel and in the manifest. Separate lossless runs restore main exactly.

This initial evidence commit contains square media only. Final profiler-free timing, full diagnostic precision audit, wide media and official Draft PR are still pending; publication of these files does not qualify or count a PR.
