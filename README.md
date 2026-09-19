# Qwen-Image 2.1 optimization: original output comparisons

Assets for [PR #13](https://github.com/mickqian/sglang/pull/13). This separate branch contains only comparison assets and does not change the PR's implementation diff.

H200, BF16, native SGLang, FlashAttention, eager, 40 steps, CFG 1, seed 42, quality=lossless.

Baseline and optimized PNGs were copied independently from the measured A2 and C2 runs, without modifying their bytes or pixels. All six pairs match exactly in PNG bytes and RGBA pixels, including alpha. See [manifest.json](manifest.json) for provenance and SHA-256 hashes.

The measured optimization commit and the PR commit contain identical production files. These examples establish equivalence for the tested inputs, not a general model-quality score.
