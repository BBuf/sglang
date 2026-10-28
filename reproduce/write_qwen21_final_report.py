"""Render the report from audited data; an optional evidence commit pins PR links."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'evidence/preflight/qwen21-gate-up-v4-final-results-summary.json').read_text())
commit=sys.argv[1] if len(sys.argv)>1 else None
def link(path,label):
    kind='blob' if '.' in Path(path).name else 'tree'
    url=f'https://github.com/BBuf/sglang/{kind}/{commit}/{path}' if commit else path
    return f'[{label}]({url})'
def img(path,label):
    url=f'https://raw.githubusercontent.com/BBuf/sglang/{commit}/{path}' if commit else path
    return f'![{label}]({url})'
text='''Qwen-Image 2.1 runs separate BF16 gate and up projections in its image MLP. Packing them into one projection reduces saved-request E2E latency by **3.16% for 512² T2I and 2.63% for 512² editing** on RTX PRO 6000 Blackwell. The original 1024² cases improve by 1.08% and 0.94%; those smaller gains are retained below.

The change shares gate/up storage while preserving checkpoint parameter names and shapes, then reuses the existing packed SiLU-multiply operator. It adds no kernel or public flag. Eligibility is limited to SM120, resident unquantized TP1 projections, BF16 batch-one targets of 1024 or 4096 tokens, and `quality="high"`. Prefixes, exact/lossless requests, LoRA wrappers, incompatible storage/layouts and `torch.compile` retain separate projections. Views are derived from parameters, with the existing `_apply` lifecycle repacking after CPU/GPU transfers.

### Sources and comparison setup

- Frozen main: `af21e2b989e482cfdce39689126562f2209d53ac`.
- Final PR head: `efcb88a201a41b1fecc516a60df0ca0a34fb6dd5`.
- Native vLLM-Omni: `817d4c17feb702289a2d2b2cbacfd594c262b7a7`.
- Checkpoint: `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3`; all 24 runtime files hash-verified.

The same physical GPU0 was used serially: RTX PRO 6000 Blackwell Server Edition, SM120, 97887 MiB, driver 595.84. GPU1 remained idle; the shared host has no exclusive lease, so ownership was checked at boundaries and during requests. Batch/concurrency 1, 50 steps, CFG 1, CPU seed 1101, identical prompts/reference image, empty negative prompt, BF16 native components with mixed-precision internals, no quantization.

SGLang uses native resident eager with cuDNN transformer attention, compile/BCG off. Omni retains its native regional compilation and decode graphs. Its default passes both 512 tasks and 1024 editing. Default 1024 T2I fails the Chinese-image gate (RGB PSNR 24.18 dB against SGLang cuDNN); the native compiled `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` variant passes and is used there. A separate valid-variant screen selected default for the other three cases. Failed variants remain archived. Equivalent native PNG compression is level 0: SGLang `output_compression=0`, Omni `100`.

SGLang uses Torch 2.14.1+cu130 / Triton 3.8 / cuDNN 9.24; Omni uses Torch 2.13+cu130 / Triton 3.7.1 / cuDNN 9.20. Dependency versions are pinned in the evidence. These are measured native stacks, not identical compiler environments.

### Final-head results

Primary E2E measures HTTP submission through receipt of complete response bytes; client decoding and artifact writes are outside the timer. Each task follows main/PR/Omni/Omni/PR/main, two fresh processes per arm. Each process first qualifies short, long, Chinese and repeated-short prompts. Then 512 uses 6 warmups + 8 measured requests; 1024 uses 3 + 6. The performance request is the frozen short prompt. Existing disk/compiler caches remain. Nothing was rerolled or filtered.

| Case | Arm | Denoise p50 s¹ | HTTP p50 / p95 s | Sequential req/s² | Peak sampled MiB³ |
|---|---|---:|---:|---:|---:|
'''
labels={'t2i512':'512² T2I','edit512':'512² edit','t2i1024':'1024² T2I','edit1024':'1024² edit'}
for case,c in summary['cases'].items():
    for arm,a in c['arms'].items():
        label={'main':'main','final':'PR','omni':'Omni casts' if a['config']=='omni_casts' else 'Omni default'}[arm]
        text+=f"| {labels[case]} | {label} | {a['separate_native_denoise_p50_seconds']:.4f} | {a['primary_p50_seconds']:.4f} / {a['primary_p95_seconds']:.4f} | {a['sequential_requests_per_second']:.6f} | {a['peak_sampled_nvml_mib']:.1f} |\n"
text+='''
¹ Separate diagnostic runs, 4 measured requests per case/arm: SGLang synchronized native stages; Omni native pipeline timer plus a metadata-only observer. No Torch profiler during stage measurements. These scopes have instrumentation overhead and are excluded from primary ratios; do not subtract them from primary HTTP time. ² Reciprocal mean latency at concurrency 1, not saturation throughput. ³ NVML sampled every 250 ms; transient peaks can be missed. p95 is descriptive over 16 requests per arm at 512 and 12 at 1024.

Ratios below are geometric means of the two temporal pairs' per-process median ratios, rather than ratios of pooled p50s.

| Case | Main / PR | E2E reduction vs main | Omni / PR | E2E reduction vs Omni |
|---|---:|---:|---:|---:|
'''
for case,c in summary['cases'].items():
    a=c['paired_comparisons']['main_over_final'];b=c['paired_comparisons']['omni_over_final']
    text+=f"| {labels[case]} | {a['geometric_mean_ratio']:.6f}× | {a['latency_reduction_percent']:.3f}% | {b['geometric_mean_ratio']:.6f}× | {b['latency_reduction_percent']:.3f}% |\n"
text+='''
Main A/A endpoint drift was at most 0.089% at 512 and 0.020% at 1024. All final main/PR process CVs and main endpoint drift meet the original 0.5% limits. **Omni fails that CV screen** in one 512 edit process (0.621%) and both 1024 T2I processes (0.553% / 1.044%). Those failures and every sample remain recorded. Both pairs favor PR; even the fastest observed Omni request is slower than the slowest PR request in each case. The separate admission records explicitly use this post-screen interpretation, not a silent threshold change. Finite-sample separation is not a population guarantee.

Cached launch-to-ready ranges were 44.03–44.04 s for SGLang at 512, 52.04–54.04 s at 1024, and 36.02–40.02 s for Omni. This excludes the four qualification requests and explicit warmups, whose individual costs are recorded in the summary. Eager has no BCG capture cost. At 1024 each SGLang process warms only 1024: a preceding 512 warmup disables the pre-existing native packed-QKV path through its bit-exact gate in both main and PR. No unrelated QKV change is included.

### Profile, cleanup and compatibility

Final representative traces show **64 separate gate/up GEMMs → 32 packed GEMMs**, plus 32 existing packed SiLU calls per denoise step. At 512 total GPU kernels fall 480 → 448; fresh 1024-only traces retain native packed QKV and fall 416 → 384. Matched complete GPU step durations, diagnostic only:

| Case | Main ms | PR ms |
|---|---:|---:|
| 512² T2I | 49.466 | 47.513 |
| 512² edit | 51.971 | 50.165 |
| 1024² T2I | 188.065 | 186.076 |
| 1024² edit | 212.939 | 210.698 |

Cleanup narrows the original broader length predicate to the two model-qualified targets and simplifies comments/docs. Two fresh-process ABBA screens retained T2I pairs at +0.1338% and +0.1688%, exceeding the fixed 0.1% cleanup screen. These are **failed screens**, not discarded warmups. Complete production AST comparison finds only the length predicate changed; it is equivalent at the tested shapes. A separately preregistered within-process diagnostic alternated the actual archived precleanup/final functions at request boundaries: 8 ABBA quartets per 512 task and 6 per 1024 task, after 8 warmups. All 144 outputs match main. Final/pre ratios are 0.999757 / 1.000049 at 512 and 1.000071 / 1.000069 at 1024; each descriptive per-case t95 upper bound is below 1.001. This supports no measurable direct cleanup penalty within that bound, while retaining the inconclusive fresh-process evidence. It does not prove zero effect or provide a familywise guarantee. These instrumented timings never enter the performance table.

Final compatibility validation covers 68 saved requests, including 12 separate profiles: high/lossless/exact/high-return, both shapes, short/long/Chinese/repeated prompts. High T2I executes one prefill plus 49 BCG replays; each profiled replay contains 32 packed SiLU calls and 33 graph launches. No serving capture occurs. Editing exceeds the native conditioning bucket and remains eager; non-high requests also remain eager and restore main pixels with no packed-MLP calls. The two-shape warmup creates 10 actual graph entries. A single cached diagnostic observed 47480 vs 39782 MiB before requests and 58.04 vs 56.04 s ready time for BCG/eager. This is compatibility and cost evidence, **not a BCG performance win**.

### Quality and real outputs

Every final saved SGLang qualification/measurement output has RGBA pixels byte-identical to its corresponding main reference. High PR/main uses fixed SSIM 0.99 / PSNR 40 dB gates; cross-framework RGB and alpha use 0.95 / 28 dB separately. Minimum final selected-Omni metrics:

| Case | RGB SSIM / PSNR dB | Alpha SSIM / PSNR dB |
|---|---:|---:|
'''
for case,c in summary['cases'].items():
    q=c['cross_framework_minimum_metrics'];r=q['rgb'];a=q['alpha']
    text+=f"| {labels[case]} | {r['ssim']:.6f} / {r['psnr_db']:.3f} | {a['ssim']:.6f} / {a['psnr_db']:.3f} |\n"
text+='''
CPU processor inputs, token IDs, reference pixels and initial noise were checked. Effective BF16 model timesteps match at 32 recorded boundaries; the native **sigma schedule** differs by at most two FP32 ULP from `numpy.exp` vs `math.exp`. This is not a bound on all solver arithmetic. T2I encoder hidden states match; edit hidden states differ before collapse. The runtime precision audit retains those differences. Final Omni native-stage observations verify BF16 scheduler input/output boundaries with autocast off. Cross-framework intermediate identity is not claimed.

Each panel below contains three actual output pairs (short, long, Chinese), main left / final PR right. Original RGBA pixels are kept at 1:1 with labels outside them; GitHub may scale the display. The source hashes, inputs and visual review are archived. Editing changes the reference red teapot to blue. These are generated results, not illustrations.

'''
for size in [512,1024]:
    for task in ['t2i','edit']:
        text+=img(f'media/{size}/{task}-{size}-main-vs-pr.png',f'Qwen-Image 2.1 {size} square {task}: main left, final PR right')+'\n\n'
text+=r'''### Validation, reproduction and limits

The real SM120 regression passed both 1024/4096-token cases: numeric/dispatch checks, original-path restoration, prefix fallback, independent storage, CPU/GPU round trip, actual dynamic LoRA weights and `torch.compile(backend="eager", fullgraph=True)` fallback. The latter checks frontend fallback, not full-model Inductor performance. Ruff 0.15.1/isort, Python/diff checks and Mint validation/broken-links passed. Final diff: 3 files, 114 additions / 7 deletions, including the regression and scoped docs.

Existing INT8 packing #44196 remains unchanged. Closed, unmerged #40918 is BF16/QKV/VAE prior art; this PR supplies a scoped BF16 path with parameter/lifecycle preservation and fresh SM120 model evidence. Latest main checked was `5cc9c933fc8acdffb5fc847d3b4824f082cd3297`: its Qwen encoder is unchanged; the DiT differs from frozen main only in the reviewed BCG attention-output allocation. Three-way textual merge checks pass for the three changed files; this is not a new benchmark or runtime test of merged main.

Use separate pinned environments with `/task` containing the archived inputs/scripts, verified checkpoint at `/task/cache/common/Qwen-Image-2.1`, main at `/task/repos/sglang`, final at `/task/candidates/qwen21-bf16-gate-up-v4`, and Omni at `/task/repos/vllm-omni`. Start one server at a time and stop only its recorded owned process before the next arm. Example 512 launches:

```bash
export QWEN_CANDIDATE_DIFF_SHA=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
# Main SGLang environment:
SGL_REPO=/task/repos/sglang sh /task/launch_sgl_qwen21_main_512.sh
# Final PR, separate process:
SGL_REPO=/task/candidates/qwen21-bf16-gate-up-v4 sh /task/launch_sgl_qwen21_v4_512.sh
# Native Omni environment:
sh /task/launch_omni_qwen21_base_native.sh
```

Against the running PR server, qualify first with a fresh output label:

```bash
python /task/submit_qwen21_candidate_qualification.py sgl \
  --label reproduction-pr-t2i512 --quality high \
  --contract /task/inputs/qwen21-base-t2i-512-workloads.json \
  --repo /task/candidates/qwen21-bf16-gate-up-v4 \
  --expected-sha efcb88a201a41b1fecc516a60df0ca0a34fb6dd5
```

Then use `bench_qwen21_serving.py` with the same arm/contract/repo/SHA, `--quality high --warmups 6 --repeats 8`, a fresh `--run-id`, and `--allowed-gpu-pids` containing only the current owned server's NVML-visible PID(s). Recreate the A/A and paired orders. For 1024 use the original non-512 contracts, `launch_sgl_qwen21_candidate.sh` for main, `launch_sgl_qwen21_v4_1024.sh` for PR, `--warmups 3 --repeats 6`, and the native casts launcher for Omni T2I only. Every exact executed command and request is archived.

Controllers contain original host/container paths and reject overwritten output IDs; adapt them to a fresh namespace, recreate quality references, and never reuse recorded PIDs. Primary runs use `SYNC_STAGE_PROFILING=0`; separate denoise launchers use synchronized native stages. Profile requests are diagnostic only.

Performance claims cover these single-GPU, single-reference, batch/concurrency-one shapes. Other lengths, GPUs, quantized checkpoints, full-model offload/LoRA/compile combinations, broader caches and multi-GPU cases are not validated performance claims. The broader frozen model matrix remains incomplete.

'''
text+='Evidence: '+', '.join([
    link('audits/preflight/qwen21-gate-up-v4-final-results-summary.json','final numbers and warmup costs'),
    link('audits/preflight/qwen21-gate-up-v4-cleanup-admission.json','cleanup admission and failed screens'),
    link('audits/preflight/qwen21-final-runtime-precision-audit.json','runtime precision'),
    link('plans','paired plans'),link('raw','raw results and exact commands'),
    link('profiles','actual traces'),link('reproduce','reproduction scripts')])+'.\n'
if commit:
    (ROOT/'evidence/qwen21-pr-body-final.md').write_text(text)
else:
    text+='''
### Archive layout

This independent evidence branch keeps all media/traces out of the code PR. `raw/` retains A/A, earlier prototype ABBA, both failed cleanup screens, within-process cleanup diagnostics, valid-variant screening, final primary and native-stage measurements. `audits/` includes quality/source/ownership checks and earlier failed variants; historical pending statuses are superseded by final audits, not rewritten. `compatibility/` and `diagnostics/` retain request transitions, replay observers and stage metadata. `media/source-images/` preserves the unique final main/PR source PNGs. `frozen-matrix-with-execution.json` retains unrun and failed cells across the broader task.

Public response exports omit duplicate base64 PNG payloads and include original response SHA256. Every original response and PNG remains in local and remote task archives. Public timing rows, requests, PNG hashes and observer logs are retained without request filtering. Binary intermediate tensors stay in the task archives; their comparison metrics/source hashes are public. `evidence-files-sha256.json` hashes every public file except itself and Git metadata. No model weights are included.
'''
    (ROOT/'artifacts/qwen21-bf16-gate-up/README.md').write_text(text)
print('Wrote', 'final PR body' if commit else 'evidence README', len(text), 'characters')
