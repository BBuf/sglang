# SANA-Video eager LN fusion: final-head evidence

This evidence branch is separate from the production code branch. It contains actual outputs, raw measurements, profiles, failures and reproduction scripts; no model weights.

- Frozen SGLang main: `af21e2b989e482cfdce39689126562f2209d53ac`.
- Final tested head: `e0d694593a65cb5b463c4011b84ee50abefa5e43`.
- Frozen vLLM-Omni: `817d4c17feb702289a2d2b2cbacfd594c262b7a7`.
- Checkpoint: `Efficient-Large-Model/SANA-Video_2B_480p_diffusers@db5f398b13ca086d09a50ce156c20527773841b1`.
- One physical RTX PRO 6000 Blackwell Server Edition GPU0, SM120; GPU1 idle. The contract is 832×480, 81 frames, 16 fps, 50 steps, CFG 6, CPU seed 1101, batch/concurrency 1. Exact prompts and encoding settings are in [the workload](inputs/sana-video-480p-t2v-workloads.json).

Read [the final measurement table](performance-table.md), [original final-primary/cleanup admission](evidence/preflight/sana-video-ln-final-primary-cleanup1-admission.json), [summary](evidence/preflight/sana-video-ln-final-summary2.json), and [frozen sequence](evidence/plans/sana-video-ln-final-primary-cleanup1.json). Every predeclared sample is retained. Primary HTTP latency, separately synchronized denoise and cached startup costs use distinct measurement scopes. Request memory is sampled with NVML every 250 ms, so the peak is an observed device-memory value.

This change enables an existing verified fusion in eager video execution. Main's BCG path already uses it. It does not establish new BCG support, a new BCG net win, or dominance over the entire frozen matrix. Chinese cross-framework PSNR fails at frames 26–28 and is excluded from primary performance admission; its outputs and failure remain public.

## Quality, execution and compatibility

- [Native cross-framework frame quality](evidence/preflight/sana-video-native-cross-quality1.json) and [runtime input audit](evidence/preflight/sana-video-runtime-input-audit1.json): token IDs/masks, noise and schedules, including numerical differences in later native computations.
- [Final compatibility admission](evidence/preflight/sana-video-ln-final-compatibility1-admission.json): 51 actual MP4s, same-quality main/PR exactness, eager/BCG and lossless/exact/high transitions, including repeated and Chinese requests.
- [Final profile attribution](evidence/preflight/sana-video-ln-final-profile1.json) and [diagnostic costs](evidence/preflight/sana-video-ln-final-diagnostic-costs2.json): actual fused kernels, BCG graph construction/replay, and scope limitations.
- [Omni synchronized stage admission](evidence/preflight/sana-video-omni-stages1-admission.json): four additional original HTTP requests, all MP4 bytes and 81 frame hashes unchanged. The existing native profiler synchronizes around `encode_prompt`, `diffuse`, `vae.decode` and `forward`; an external observer only adds method targets. One startup forward is separated by the actual application-startup marker, and the four serving groups are each matched to their HTTP 200 response. The original collector’s five-versus-four count failure is retained; an offline audit corrected the classification without another GPU run. The first real short request is retained and excluded from the representative steady diagnostic, leaving n=1 per English prompt.
- [Numerical/CPU test receipts](evidence/block-runs/sana-video-ln-final-tests2/results.json). The initial hidden-CUDA CPU-import failure is retained in `evidence/block-runs/sana-video-ln-final-tests1`; the unchanged CPU-tensor diagnostic passed with CUDA visible for import. It is not a GPU compile benchmark.

Actual main/final frames 0/40/80 at 0/2.5/5 seconds, with labels identifying the configuration and commits:

![Short prompt: actual main and final outputs](media/short-main-vs-pr.png)

![Long prompt: actual main and final outputs](media/long-main-vs-pr.png)

![Chinese prompt: actual main and final outputs; cross-framework failure retained](media/chinese-main-vs-pr.png)

Original MP4s: [short main](media/videos/short-main.mp4), [short PR](media/videos/short-pr.mp4), [long main](media/videos/long-main.mp4), [long PR](media/videos/long-pr.mp4), [Chinese main](media/videos/chinese-main.mp4), [Chinese PR](media/videos/chinese-pr.mp4). [Video identities](actual-main-pr-videos.json) and [manual representative-frame review](media/visual-review.json) preserve the origin and observed baseline artifacts. No full-motion review is claimed.

## Reproduction and provenance

The recorded host root is `/data/xyzhang/diffusion-rtx6000-20261026`, bind-mounted as `/task` in the two owned containers. These scripts intentionally assert exact sources, payload hashes, expected output hashes, idle GPUs and new result directories. Use a fresh reproduction directory and equivalent mounts; do not overwrite the archived experiment or remove guards to make a changed setup pass.

[Container image identities](evidence/preflight/sana-video-container-image-identities1.json), [complete SGLang install record](sgl-pip-freeze.txt), [complete Omni install record](omni-pip-freeze.txt), and [dependency provenance](evidence/preflight/sana-video-dependency-provenance1.json) disclose the actual environments. The Omni base was a local image; public availability of that image or local wheel paths is not claimed. Ordinary version pins are also provided, but do not replace the complete records. SGLang launchers select the exact worktree with `PYTHONPATH`, overriding the editable baseline installation.

Restore the pinned checkpoint from its original repository or use `reproduce/download_pinned_modelscope_ranges.py` with `evidence/plans/sana-video-480p-download1.json`. The downloader requires the original HF hashes and sends no credentials to the public mirror. Preserve the actual native model configuration. Source checkouts and the V1 worktree are identified in the frozen plan; `evidence/preflight/sana-video-eager-ln-v1.patch` is the exact pre-cleanup patch.

The stage supplement is reproduced with `launch_omni_sana_video_stages1.sh` and the external observer installed as `/task/sana-video-stage-observer1/sitecustomize.py`; its frozen plan records every payload hash. `admit_and_archive_sana_video_omni_stages1.py` supports `DIFFUSION_TASK_ROOT=/task` inside the container. It preserves the original collector error and verifies request/log boundaries rather than rerunning the GPU.

The complete launch and client argument lists are recorded in every `evidence/block-runs/*/metadata.json`; the tested launchers and runners are in `reproduce/`. With the recorded directory/container layout and verified prerequisites:

```bash
# Inside the pinned SGLang container; exact final worktree selected by the launcher:
sh /task/launch_sgl_sana_video_final_primary1.sh
# In a separate client session, inspect the unchanged benchmark arguments:
python3 /task/bench_sana_video_serving1.py --help

# Host-side fixed 12-process sequence, after preparing its required reference metadata:
python3 /data/xyzhang/diffusion-rtx6000-20261026/run_sana_video_blocks1.py \
  /data/xyzhang/diffusion-rtx6000-20261026/sana-video-ln-final-primary-cleanup1.json
```

The serial runner manages server shutdown between blocks; do not leave the example manual server running when starting it. Its full plan preserves main/V1/final/Omni order and prompt reversal. For exact layout restoration, local evidence paths `evidence/qualification`, `evidence/benchmarks`, `evidence/diagnostics` and `evidence/profiles` correspond to the host's `outputs/` subdirectories; `evidence/block-runs` corresponds to host `block-runs`. Install `reproduce/` scripts at the host root, the workload under `inputs/`, and plans at the root when their runners expect that location. New measurements must use fresh output locations; historical reference results and quality audits remain immutable.

Large duplicate media are stored once under `media/by-sha256`; [the original-path map](original-path-map.json) reconstructs their evidence locations. Large logs are losslessly gzipped, with original hashes in the same map. Binary tensor snapshots and Git transfer bundles remain in fully verified local archives; [omissions](omitted-binary-files.json) identify every such file. [Archive manifests](archive-manifests) and `evidence-files-sha256.json` record hashes. Traces, media and historical failed attempts are evidence, not production code changes.
