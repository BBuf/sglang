"""Build a reviewable PR body from admitted final-head results and committed media."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--artifact-commit', required=True)
args = parser.parse_args()
assert re.fullmatch('[0-9a-f]{40}', args.artifact_commit)
root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
artifacts = root/'artifacts/sana-video-depthwise-glu'
head = 'f248594caca5c533c94612b6848ff3512ed98e08'
base = 'af21e2b989e482cfdce39689126562f2209d53ac'
omni = '817d4c17feb702289a2d2b2cbacfd594c262b7a7'


def read(name):
    return json.loads((evidence/name).read_text())


assert subprocess.check_output(['git', '-C', str(artifacts), 'rev-parse', 'HEAD'], text=True).strip() == args.artifact_commit
assert not subprocess.check_output(['git', '-C', str(artifacts), 'status', '--porcelain'])
primary = read('preflight/sana-video-depthwise-final-primary-cleanup1-admission.json')
raw_audit = read('preflight/sana-video-depthwise-final-primary1-raw-audit.json')
assert raw_audit['passed'] and raw_audit['raw_request_count'] == 96
compat = read('preflight/sana-video-depthwise-final-compatibility1-admission.json')
profiles = read('preflight/sana-video-depthwise-final-profile1.json')
tests = read('preflight/sana-video-depthwise-final-tests1-verification.json')
summary = read('preflight/sana-video-depthwise-final-summary1.json')
aa = read('preflight/sana-video-depthwise-main-eager-aa1-admission.json')
cross_quality = read('preflight/sana-video-native-cross-quality1.json')
worst = read('media/sana-video-native-cross-worst1/manifest.json')
worst_review = read('media/sana-video-native-cross-worst1/visual-review.json')
merge = read('preflight/sana-video-depthwise-main-merge-preflight2.json')
assert merge['final_head'] == head and merge['frozen_base'] == base
assert not merge['merge_conflicts'] and merge['exact_change_content_preserved']
assert worst_review['passed'] and not worst_review['quality_passed']
assert worst_review['panel_sha256'] == worst['panel_sha256']
assert hashlib.sha256((artifacts/worst['panel']).read_bytes()).hexdigest() == worst['panel_sha256']
assert all(item['passed'] for item in [primary, compat, profiles, tests, aa])
assert compat['source_sha'] == tests['source_sha'] == summary['source_sha'] == head
assert primary['all_saved_MP4s_exact_qualified_native'] and compat['raw_output_count'] == 51
media = read('media/sana-video-depthwise-final-f248594c/manifest.json')
visual = read('media/sana-video-depthwise-final-f248594c/visual-review.json')
assert visual['passed'] and visual['source_sha'] == head and len(media['rows']) == 3
for row in media['rows']:
    panel = root/row['panel']
    assert hashlib.sha256(panel.read_bytes()).hexdigest() == row['panel_sha256']
    assert (artifacts/'media'/panel.name).read_bytes() == panel.read_bytes()
    assert row['MP4_exact'] and row['all81frames_exact']
videos = json.loads((artifacts/'actual-main-pr-videos.json').read_text())
assert len(videos) == 6
for video in videos:
    assert hashlib.sha256((artifacts/video['public_path']).read_bytes()).hexdigest() == video['sha256']

url = f'https://github.com/BBuf/sglang/blob/{args.artifact_commit}'
raw = f'https://raw.githubusercontent.com/BBuf/sglang/{args.artifact_commit}'


def link(label, path):
    assert (artifacts/path).exists(), path
    return f'[{label}]({url}/{path})'


body = [
    '## Motivation and change',
    '',
    'SANA-Video applies a BF16 depthwise 3×3 convolution followed by bias and GLU to large channel-last activations. The tested `[21, 13440, 30, 52]` activation materializes an approximately 840 MiB convolution intermediate. A complete CFG iteration contains 40 depthwise convolutions and 40 separate GLU kernels in the frozen baseline.',
    '',
    'This PR fuses those operations in one Triton kernel. Four adjacent output rows reuse input loads. It preserves row-major FP32 FMA accumulation and the native BF16 convolution, bias and SiLU rounding boundaries. Dispatch uses the existing per-signature exactness gate and is restricted to SM120 inference with the supported large BF16 channel-last convolution. Unseen signatures during capture, unsupported convolution/layout settings, gradients, compilation and disabled cuDNN retain the reference path. A failed exactness verification disables the fusion. SANA-WM remains unchanged because its biased-convolution arithmetic differs.',
    '',
    'The implementation lives in the KDA backend and is exposed through the existing lazy diffusion facade. This independent change does not include the separate eager-LN optimization. Existing RoPE and bias-SiLU fusions remain in place.',
    '',
    '## Final-head benchmark',
    '',
    f'Frozen SGLang main: `{base}`. Final tested PR head: `{head}`. Frozen vLLM-Omni: `{omni}`.',
    'The '+link('independent raw audit', 'evidence/preflight/sana-video-depthwise-final-primary1-raw-audit.json')+' checks every saved request, native MP4, runtime commit and GPU ownership sample and recomputes the original statistics. It does not change the frozen samples or thresholds.',
    '',
    'One NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, 97,887 MiB, 600 W), GPU 0 only; GPU 1 idle. Checkpoint `Efficient-Large-Model/SANA-Video_2B_480p_diffusers@db5f398b13ca086d09a50ce156c20527773841b1`. Native SGLang and Omni backends; BF16 DiT/text encoder, FP32 VAE/noise/solver. Workload: 832×480, 81 frames at 16 fps, 50 steps, CFG 6, CPU seed 1101, empty negative prompt, 300-token cap, batch/concurrency 1.',
    '',
    (evidence/'sana-video-depthwise-final-performance-table1.md').read_text().strip(),
    '',
    'The prospectively frozen order is M–V1–F–O–V1–F–F–V1–O–F–V1–M, with one excluded warmup and three measured requests per English prompt per process. Prompt order reverses in the second half. Main/Omni have n=6 per prompt and V1/final n=12. HTTP E2E includes request submission, native polling where applicable, encoding and complete MP4 transfer, but excludes validation and disk writes. The primary SGLang process has no profiler, observer or added stage synchronization. All original samples and warmups are retained.',
    '',
    f'Fresh main A/A maximum noise was {aa["max_noise_percent"]:.8f}%. Cleanup mean regression must be no greater than this noise; the four-pair log-t 95% upper regression bound must also remain below 0.1%. The final table includes the complete paired results and uncertainty. Two main/final and Omni/final process pairs limit interval precision.',
    '',
    '## Actual execution and compatibility',
    '',
    '| Configuration | GPU kernels / CFG iteration | Kernel sum (ms) | Depthwise+GLU | Native depthwise | Separate GLU | Graph launches |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for row in profiles['rows']:
    c = row['counts']
    body.append(f'| {row["arm"]} | {row["kernel_count"]} | {row["kernel_sum_ms"]:.3f} | {c["depthwise_glu"]} | {c["native_depthwise"]} | {c["fused_conv_glu"]} | {row["graph_launches"]} |')
body.extend([
    '',
    'Profiles are separate synchronized diagnostics. CPU `ProfilerStep#3` launch correlations identify a complete CFG iteration and the actual kernels inside graph launches. The final eager path retains all 82 ATen LayerNorm calls; the existing BCG path uses its 82 fused LN calls. Both retain 40 RoPE and 40 bias-SiLU calls. Kernel sums are not primary serving latency.',
    '',
    '| Configuration | Serving requests | Model replays | Serving captures | Warmup graph captures | Capture time sum (s) | Launch-to-ready (s) | Full diagnostic process peak (MiB) |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
])
for row in compat['rows']:
    captures = row['actual_warmup_captures']
    body.append(f'| {row["configuration"]} | {row["serving_requests"]} | {row["serving_replay_calls"]} | {row["serving_capture_calls"]} | {len(captures)} | {sum(c["wall_seconds"] for c in captures):.3f} | {row["launch_to_ready_seconds"]:.3f} | {row["diagnostic_full_process_peak_gpu0_memory_mib"]:.0f} |')
body.extend([
    '',
    'The compatibility sequence switches short/long/Chinese prompts and lossless/exact/high modes, including repeated requests and return to lossless. Exact/high use uncaptured quality-specific eager runners; returning to lossless reuses the graph without serving capture. Final BCG profiles place all 40 new depthwise+GLU kernels inside graph replay. These results establish compatibility with existing BCG, not a new beneficial BCG scenario. Full-process diagnostic memory includes loading, warmup, quality switching and profiling; it does not isolate serving-only BCG overhead. Launch/capture observations are cached individual runs, not cold-start distributions.',
    '',
    '## Quality and actual generated results',
    '',
    'All 96 final primary MP4s match their qualified native-arm reference hashes. The 51 compatibility MP4s match the same-quality main outputs, including every decoded frame. Native input auditing checks actual token IDs/masks, CPU noise and schedules. English cross-framework workloads pass the frozen quality criteria. Chinese cross-framework PSNR fails at frames 26–28 (minimum 23.5768 dB below 24 dB); its outputs and failure are retained and excluded from primary timing admission. Main/PR Chinese outputs remain exact.',
    '',
    'Native SGLang main versus native Omni, with all 81 decoded frames evaluated per request. Frozen admission requires every frame to have SSIM ≥ 0.92 and PSNR ≥ 24 dB; passing an average alone is insufficient.',
    '',
    '| Request | SSIM mean / minimum | PSNR mean / minimum (dB) | All-frame admission |',
    '|---|---:|---:|---|',
])
for row in cross_quality['rows']:
    assert len(row['frames']) == 81
    result = 'pass' if row['all_frames_pass'] else 'FAIL; excluded from primary timing'
    body.append(f'| {row["request"]} | {row["mean_ssim"]:.6f} / {row["min_ssim"]:.6f} | {row["mean_finite_psnr_db"]:.4f} / {row["min_psnr_db"]:.4f} | {result} |')
body.extend([
    '',
    'The panels below show actual main/final frames 0/40/80 at 0/2.5/5 seconds with identical prompts and generation settings. Representative frames were manually reviewed; full-motion review is not claimed. Shared model artifacts remain visible: an extra teapot in the short prompt, two spouts in the long prompt, and malformed stacked lid/cup geometry in the Chinese prompt.',
])
for row in media['rows']:
    prompt = row['prompt']
    body.extend(['', f'![Actual {prompt} main and PR frames]({raw}/media/{prompt}-main-vs-pr.png)', '',
                 f'Original videos: [main]({raw}/media/videos/{prompt}-main.mp4) · [PR]({raw}/media/videos/{prompt}-pr.mp4).'])
body.extend([
    '',
    'The retained cross-framework failure is shown separately below: original main/Omni Chinese frames 26–28 at 1.625–1.750 seconds. Frame 27 is the worst PSNR sample. Differences in vessel contour, steam and cloth remain visible. These frames were exported with the original common decoder and checked against all 81 original frame hashes per video; no model was re-run.',
    '',
    f'![Retained Chinese cross-framework worst frames]({raw}/{worst["panel"]})',
    '',
    'Quality details: '+link('all-frame metrics', 'evidence/preflight/sana-video-native-cross-quality1.json')+', '+link('worst-frame provenance', 'evidence/media/sana-video-native-cross-worst1/manifest.json')+', '+link('worst-frame visual review', 'evidence/media/sana-video-native-cross-worst1/visual-review.json')+'. The initial local decoder failed the RGB-hash guard and produced no admitted media; its failure is retained.',
    '',
    '## Cleanup, validation and reproduction',
    '',
    'The complete five-file diff was reviewed against BBuf cleanup examples #42391, #40265 and #39293. Cleanup removes the unused row-count parameter, dead constant branches and redundant additions while preserving the accumulation/rounding order; tests extend the existing regression files. The scoped pre-commit checks and diff whitespace check passed. Final runtime validation passed 135 namespace/dispatch tests, 11 numerical/gate tests (six new cases) and an actual CPU fullgraph FX fallback check with changed inputs and bias. The CPU diagnostic does not establish compiled GPU performance, and local checks do not imply upstream GPU CI passed.',
    '',
    'The first per-pixel fused kernels were slower, and five tested FMA orders failed exactness; these experiments remain archived. An initial qualification preparation failure occurred before GPU work and was preserved with its worktree-index repair. No failed or slower measurements were silently removed.',
    '',
    'The candidate uses a separate RTX6000 checkout/branch. '+link('All five changed files are registered', 'shared-file-registration.json')+f' for cross-task coordination. A local virtual merge with main `{merge["latest_main"]}` has no conflicts and preserves the exact changed content; '+link('the merge receipt', 'evidence/preflight/sana-video-depthwise-main-merge-preflight2.json')+' records this source check. The merged tree was not benchmarked; all measured source SHAs above remain frozen.',
    '',
    'Reproduction: '+link('README', 'README.md')+', '+link('frozen final sequence', 'evidence/plans/sana-video-depthwise-final-primary-cleanup1.json')+', '+link('all raw launch/client records', 'evidence/block-runs')+', '+link('final primary/cleanup admission', 'evidence/preflight/sana-video-depthwise-final-primary-cleanup1-admission.json')+', '+link('compatibility', 'evidence/preflight/sana-video-depthwise-final-compatibility1-admission.json')+', '+link('actual profiles', 'evidence/preflight/sana-video-depthwise-final-profile1.json')+', '+link('runtime input audit', 'evidence/preflight/sana-video-runtime-input-audit1.json')+', '+link('video identities', 'actual-main-pr-videos.json')+'.',
    '',
    '```bash',
    '# Inside the pinned SGLang container, using the exact final worktree:',
    'sh /task/launch_sgl_sana_video_depthwise_final_primary1.sh',
    '# Host-side paired runner after restoring the documented prerequisites:',
    'python3 /data/xyzhang/diffusion-rtx6000-20261026/run_sana_video_blocks1.py \\',
    '  /data/xyzhang/diffusion-rtx6000-20261026/sana-video-depthwise-final-primary-cleanup1.json',
    '```',
    '',
    'Stop the manual example server before starting the serial runner. The README records environment versions, image/local-wheel provenance, guards, archive hashes and media path mappings. Evidence lives in a separate branch; weights, traces and media are outside the production diff.',
    '',
    'Scope is the native 480p, single-GPU workload above. Other resolutions, batch/concurrency, GPU generations, GPU compile and quantization are unrun. The frozen matrix retains 720p/I2V gaps, failures and slower cells; these measurements do not claim full-matrix dominance. Reported runtime evidence uses the frozen af21 baseline, not a re-run of subsequent common-runner changes.',
])
output = evidence/'sana-video-depthwise-pr-body-draft.md'
assert not output.exists()
output.write_text('\n'.join(body)+'\n')
print(output)
