"""Report the complete primary rerun and separate diagnostics without changing admission."""
import hashlib,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'evidence'
read=lambda p:json.loads(p.read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head='188169fa0db0bb2d524d56a97184590f948cca07'
lines=['# Qwen PR47090: current-head cleanup revalidation incomplete','',
'Current head `188169fa0db0bb2d524d56a97184590f948cca07` passes the frozen 512 gates but fails the frozen 1024 cleanup gate in both T2I and editing. All samples and original thresholds are retained. This report does not qualify the current head. The original PR report remains historical evidence for `efcb88a201a41b1fecc516a60df0ca0a34fb6dd5`.', '',
'Frozen SGLang main `af21e2b989e482cfdce39689126562f2209d53ac`; native vLLM-Omni `817d4c17feb702289a2d2b2cbacfd594c262b7a7`; checkpoint `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3`, all 24 runtime files hash verified. Same physical RTX PRO 6000 Blackwell Server GPU0 (GPU-1565b45e-a702-2015-9776-61f68c36fd59), GPU1 idle, driver 595.84. SGLang Torch2.14.1+cu130, Omni Torch2.13.0+cu130. BF16 native components with recorded internal mixed precision, CPU seed1101, 50steps, CFG1, batch/concurrency1, fixed inputs and RGBA PNG semantics.', '',
'New primary runs use native resident SGLang eager, SYNC0, without a profiler or observer. Omni retains its native regional compilation and decode graphs. 512 uses native default; 1024 T2I uses the previously quality-qualified native `emulate_precision_casts`, while 1024 editing uses native default. The earlier default1024 T2I quality failure remains in the original PR evidence; it is not removed or counted as a win.', '',
'Each resolution has a fresh two-process main A/A followed by the frozen paired primary schedule. Per task: two main and two native Omni processes (16 measured requests each), four original and four current processes (32 each); each process/task also retains four qualification requests, six excluded warmups and eight measured short requests. The second half reverses task order. Total primary: 12 processes/432 PNGs at512 and14 processes/432 PNGs at1024. A/A adds72 PNGs per resolution. Process pairs, not individual requests, are the independent units in paired confidence intervals.', '',
'## Primary E2E and memory','',
'| Shape/task | Arm | n | Mean s | p50 s | p95 s | Serial response/s | Peak NVML MiB |',
'|---|---|---:|---:|---:|---:|---:|---:|']
summary={};quality={}
for size in [512,1024]:
 name=f'qwen21-current-head-{size}-primary1';a=read(E/'preflight'/(name+'-admission.json'));p=read(E/'plans'/(name+'.json'))
 assert a['source_sha']==head and a['passed']==(size==512) and a['all_samples_retained'] and a['source_and_ownership_audit_passed']
 summary[str(size)]=a
 for task,c in a['cases'].items():
  for arm,v in c['arms'].items():lines.append(f'| {size} {task} | {arm} | {v["n"]} | {v["mean_seconds"]:.9f} | {v["p50_seconds"]:.9f} | {v["p95_seconds"]:.9f} | {v["serial_response_rate_per_second"]:.6f} | {v["peak_gpu_memory_bytes"]/2**20:.0f} |')
  arm=c['omni_configuration'];i=p['order'].index(arm);q=E/'qualification'/f'omni-qwen21-base-{task}-{name}-{i:02d}-{arm}-high-quality.json';d=read(q);assert d['all_pass']
  rows=[x for x in d['results'] if x['reference'].startswith('sgl-') and not x['prompt'].startswith('03-')];assert len(rows)==3
  main=read(E/'qualification'/f'sgl-qwen21-base-{task}-{name}-00-main-high/results.json')
  assert all(x['reference_sha256']==main[j]['sha256'] for j,x in enumerate(rows))
  quality[f'{size}-{task}']=dict(source=str(q.relative_to(R)),source_sha256=digest(q),rgb_ssim_min=min(x['rgb']['ssim'] for x in rows),rgb_psnr_min=min(x['rgb']['psnr_db'] for x in rows if x['rgb']['psnr_db'] is not None),alpha_ssim_min=min(x['alpha']['ssim'] for x in rows),alpha_psnr_min=min(x['alpha']['psnr_db'] for x in rows if x['alpha']['psnr_db'] is not None),samples=rows)
lines+=['','Serial response rate is reciprocal mean serial latency, not a saturation-throughput result. Peak NVML includes process/device residency; it is not tensor allocator size.','',
'## Fixed gates and uncertainty','',
'| Shape/task | E2E reduction vs main | Omni/current mean ratio | Cleanup mean change | Fresh AA noise | Cleanup t95 upper regression | Primary / cleanup |',
'|---|---:|---:|---:|---:|---:|---|']
for size,a in summary.items():
 for task,c in a['cases'].items():lines.append(f'| {size} {task} | {c["final_vs_main_latency_reduction_percent"]:.6f}% | {c["omni_over_current_speedup"]:.9f}x | {c["cleanup_mean_latency_change_percent"]:+.9f}% | {c["AAmax_noise_percent"]:.9f}% | {c["cleanup_t95_regression_upper_percent"]:.9f}% | pass / '+('pass' if c['cleanup_passed'] else '**FAIL**')+' |')
lines+=['','Cleanup requires mean regression no greater than fresh AA noise AND a four-pair log-t95 regression upper bound below0.1%. Both1024 means exceed AA noise; T2I also fails the upper-bound test. The two1024 cleanup intervals include1: this does not prove a nonzero population regression or a source-code cause, and does not satisfy the frozen no-regression assurance. No unchanged primary rerun or relaxed gate was used.','',
'| Shape/task | Ratio | Process pairs | Geomean | Paired log-t95 interval |','|---|---|---:|---:|---|']
for size,a in summary.items():
 for task,c in a['cases'].items():
  for label,pair in c['pairs'].items():
   lo,hi=pair['log_t95_interval'];lines.append(f'| {size} {task} | {label} | {len(pair["ratios"])} | {pair["geometric_mean"]:.9f} | [{lo:.9f}, {hi:.9f}] |')
lines+=['','Both512 main/current intervals include1 despite their frozen gates passing; only two process pairs support these intervals. Do not claim all main/current intervals exclude1. Both1024 main/current and all four Omni/current intervals exceed1. Independent NumPy/SciPy reviews recomputed all statistics from actual raw results without the admission calculator.','',
'## Startup and excluded warmups','',
'| Shape/task | Arm | Launch-to-ready range s | Sum of six warmup requests per process, range s |','|---|---|---:|---:|']
for size,a in summary.items():
 for task,blocks in a['blocks'].items():
  for arm in a['cases'][task]['arms']:
   group=[x for x in blocks if x['config']==arm];ready=[x['launch_to_ready_seconds'] for x in group];warm=[sum(x['warmup_http_seconds']) for x in group]
   lines.append(f'| {size} {task} | {arm} | {min(ready):.3f}–{max(ready):.3f} | {min(warm):.3f}–{max(warm):.3f} |')
lines+=['','Readiness includes native startup/warmup and, for Omni, native compilation/graph setup; these costs are not included in primary saved-request E2E. The same server may handle both tasks; per-task startup entries are not additive.','',
'## Quality','',
'The twelve displayed fresh main/current primary image pairs have identical PNG bytes and RGBA pixels (RGB/alpha SSIM1, PSNR∞). Three distinct prompts per task/shape include short, long and Chinese text; repeated short requests are retained. Pixel equality does not establish semantic perfection. Cross-framework RGB and alpha are assessed separately against0.95 SSIM/28dB thresholds. Below are componentwise minima over the three distinct prompts in the first fresh native Omni process; the full per-prompt records identify potentially different worst samples.','',
'| Shape/task | RGB min SSIM | RGB min PSNR dB | Alpha min SSIM | Alpha min PSNR dB |','|---|---:|---:|---:|---:|']
for name,q in quality.items():lines.append(f'| {name} | {q["rgb_ssim_min"]:.9f} | {q["rgb_psnr_min"]:.6f} | {q["alpha_ssim_min"]:.9f} | {q["alpha_psnr_min"]:.6f} |')
f=read(E/'profiles/qwen21-cleanup-1024-forensics1-profile-review.json');assert f['all_gpu_sequences_match'] and f['all_cpu_sequences_match'] and f['all_expected_packed_dispatch']
lines+=['','## Separate failure investigation','',
'The complete original→current source diff is two files,16 insertions/12 deletions. Production changes rename the enable flags and pass one outer block flag by keyword instead of position. Arithmetic, kernels and weight storage are unchanged. The actual SM120 packing/restoration/LoRA regression passed at1024 and4096 tokens,2 tests/0 skips. This does not establish full-model LoRA or GPU compile performance.', '',
'A separately frozen original/current1024 native high run completed two fresh processes,44 PNGs and four Torch traces, all hash verified. Each task/source retains four qualification requests, two excluded warmups, four SYNC1 diagnostic requests and one separate profiled request. All44 images equal corresponding primary main RGBA outputs. Three complete interior profile steps (#2,#3,#4) per trace have identical GPU kernel sequence/launch geometry, CPU operator sequence/shapes and CUDA runtime counts:384 kernels,32 packed gate/up GEMMs and32 packed SiLU kernels per step,zero graph launches. Original/current sampled peaks are identical within this diagnostic (39339884544B T2I;45669089280B edit). This finds no operator/storage regression mechanism; Python argument-binding overhead is not isolated by these traces.', '',
'| Task | Source | SYNC1 denoise p50 s (4 diagnostic requests) | Profile GPU steps2/3/4 ms |','|---|---|---:|---|']
for task,c in f['cases'].items():
 for arm,v in c['arms'].items():lines.append(f'| {task} | {arm} | {v["synchronized_denoise_p50_seconds"]:.9f} | '+', '.join(f'{s["gpu_step_ms"]:.6f}' for s in v['steps'].values())+' |')
lines+=['','These timings are diagnostic, one process per source, and cannot replace primary E2E or prove a performance repair. New resolution-specific main/current stage diagnostics and high→lossless→exact→high/BCG compatibility plans were not run after primary failure. No current-head BCG net benefit, new scenario or complete compatibility validation is claimed. No source patch is justified solely by matching traces.','',
'## Reproduction and archive','',
'Use the pinned commits/checkpoint and an isolated owned GPU0, with the exact native launchers, contracts, warmups, process ordering and pair indices in `evidence/plans/`. `run_qwen21_current_head_blocks1.py` runs each AA or primary plan once; archive/verify/audit AA before primary. Preserve SYNC0 for primary. The separately named `qwen21-cleanup-1024-forensics1` plan uses the unmodified generic diagnostic runner with SYNC1 and explicitly pins the failed primary receipt. Do not treat it as an admission retry. Local root-level remote scripts map to `scripts/` and archived `evidence/reproduction/` paths; remote root is `/data/xyzhang/diffusion-rtx6000-20261026`, mounted `/task`.', '',
'Complete archives remain local and are identified by public per-member hashes; public PNGs are deduplicated by content hash, complete raw numerical results/requests/telemetry are retained, response copies omit repeated base64 data with original hashes recorded, and traces are published separately from production code. All historical failures remain in the original PR evidence. The full frozen81-cell matrix is not claimed complete.']
out=E/'qwen21-current-head-failure-report1.md';assert not out.exists();out.write_text('\n'.join(lines)+'\n')
qpath=E/'preflight/qwen21-current-head-failure-quality1.json';assert not qpath.exists();qpath.write_text(json.dumps(dict(source_sha=head,cases=quality),indent=2)+'\n')
print(json.dumps(dict(report=str(out),sha256=digest(out),quality=str(qpath),primary_1024_passed=False)))
