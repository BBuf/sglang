# MOSS packed QKV validation media

Actual images generated on one RTX PRO 6000 Blackwell Server Edition, using MOSS-Image-Base at checkpoint revision `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3`.

- SGLang frozen main: `af21e2b989e482cfdce39689126562f2209d53ac`.
- Candidate: `d752066eaa6f5091fa4c8daa2ba1d5d43a3928cd`.
- 1024x1024, 50 steps, CPU generator seed 1101, CFG 3, flow shift 5, skip layer 22 with SLG window 0.01-1.0, batch/concurrency 1.
- Both use native eager high quality and transformer cuDNN SDPA, BF16 DiT/text encoder and FP32 VAE/solver.

Each complete source image is pasted at 1:1 pixel scale, unchanged, with captions outside the source bounds. `media-manifest.json` records original and composed SHA256 hashes. These are real serving outputs; no images were synthesized or retouched for this comparison. Main/PR high outputs are not pixel-identical. Three prompt numeric checks and a visual review are local qualification evidence, not a broad perceptual-equivalence claim. The exact inputs are in `moss-workloads.json`.

This branch holds evidence only and is independent of the source PR branch. Benchmark analysis is pending the complete predeclared protocol; no performance claim is made by this media upload.

![Short prompt](00-image-short-main-vs-pr.png)
![Long prompt, worst main/PR numeric sample](01-image-long-main-vs-pr.png)
![Chinese text](02-image-text-main-vs-pr.png)
