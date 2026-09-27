## Motivation and changes

Speed up SANA-Video's residual-gate add on H200 using the **KDA-1.5 + Codex CLI + Kimi K3** candidate. Its transposed residual layout makes the original CUDA implementation expensive; the Triton path tiles along the physical stride-1 token dimension and preserves the original multiply-round-add semantics, output strides, guards, custom-op/fake interface and fallback.

**This PR now changes one production file and retains one kernel family: `residual_gate_add`.** The other 39 families from the earlier version have been removed entirely: any family with a measured regression, or without established material end-to-end benefit, stays at the upstream implementation. No per-shape salvage was used. Existing repository tests are unchanged, and the `kernel-design-agents` coauthor is retained.

Baseline: `c842d32b0ab45753178c5f6420662d7812766aca`. Candidate: `76aa0c6c8db3efd275a250a4b9aa07fd13b23a5a`. All results below were freshly measured with this isolated one-file candidate, so the E2E results are attributable to the retained change. The earlier 40-family combined results are not used as evidence for this revision.

## Kernel validation

NVIDIA H200; three independent processes; CUPTI cold-L2 timing. Speedup is baseline latency / candidate latency. Every original frozen case remains included; two additional actual SANA-Video layouts are reported separately. All measured ratios are at least 1; this is a statement about the tested cases, not a guarantee for arbitrary inputs.

| Workloads | Cases | Three geometric-mean speedups | Worst ratio across all repeats |
|---|---:|---|---:|
| Original frozen suite | 33 | 1.2702× / 1.2686× / 1.2698× | 1.0017× |
| SANA-Video captured layouts | 2 | 2.5418× / 2.5434× / 2.5416× | 2.5385× |

SANA layout: BF16 `[1, 32760, 2240]`, transposed residual/output, contiguous update, broadcast gate with the two captured storage offsets. Latency is approximately **331 μs → 130 μs**. Captured-layout correctness and changed-input prepared calls pass. **35 existing residual-gate tests pass**, including dtype/layout/offset, compile, graph and fallback coverage.

[Raw kernel results](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/kernel-benchmarks.json) · [Existing tests](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/existing-tests.log) · [KDA provenance](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/provenance.json) · [All 40 inclusion/removal decisions](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/selection.json)

## Isolated end-to-end results

Model: `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`, revision `db5f398b13ca086d09a50ce156c20527773841b1`; one H200; 832×480, 81 frames, 50 steps, guidance 6, seed 42. Prompt: `A curious raccoon walks through a sunlit forest. motion score: 30.` Eager native SGLang backend; fixed settings and pinned checkpoint in both arms.

Each quality runs two independent **A–B–B–A groups**, eight full saved requests total. Each cell compares the mean of two baseline requests with the mean of two candidate requests. Saved E2E includes scheduler wait and output saving; loading and separately captured profiling runs are excluded. Both groups must improve by at least 1.5% to qualify.

| Quality | Saved E2E group 1 | Saved E2E group 2 | Actual before/after |
|---|---|---|---|
| lossless | 38.840 → 38.190 s (**−1.67%**) | 38.810 → 38.155 s (**−1.69%**) | ![Actual before/after](https://raw.githubusercontent.com/BBuf/sglang/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/lossless/comparison.gif) |
| high | 36.300 → 35.670 s (**−1.74%**) | 36.295 → 35.700 s (**−1.64%**) | ![Actual before/after](https://raw.githubusercontent.com/BBuf/sglang/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/high/comparison.gif) |

All eight generated videos per quality are **file-byte and decoded-pixel exact**. GIF previews show baseline left and candidate right. Separate profiles confirm the changed `_rga_transposed` device branch executes in the real request.

- **lossless:** [baseline MP4](https://raw.githubusercontent.com/BBuf/sglang/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/lossless/baseline.mp4), [candidate MP4](https://raw.githubusercontent.com/BBuf/sglang/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/lossless/candidate.mp4), [eight-run timing](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/lossless/abba.json), [exact-output audit](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/lossless/media-audit.json), [device dispatch](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/lossless/device-dispatch.json).
- **high:** [baseline MP4](https://raw.githubusercontent.com/BBuf/sglang/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/high/baseline.mp4), [candidate MP4](https://raw.githubusercontent.com/BBuf/sglang/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/high/candidate.mp4), [eight-run timing](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/high/abba.json), [exact-output audit](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/high/media-audit.json), [device dispatch](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/models/high/device-dispatch.json).

These claims cover the listed H200 workloads and this exact PR head. No end-to-end gain is claimed for other models. Further model expansion and the previously paused optimization jobs remain paused. [Reproducible source patch](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/source.patch) · [Artifact hashes](https://github.com/BBuf/sglang/blob/codex/kda-h200-media-20260926/diffusion-prs/kda-residual-gate-h200-20260927/artifact-manifest.json).
