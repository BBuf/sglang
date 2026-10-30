"""Publishable failure evidence; this deliberately preserves passed=false."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path
r=Path(__file__).resolve().parents[1];e=r/'evidence';d=r/'artifacts/joy-pr47022-square-cleanup-failed-b08c24e9'
read=lambda p:json.loads(p.read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run='joy-current-head-square-primary1';a=read(e/'preflight'/(run+'-admission.json'));assert not a['passed'] and not a['cleanup_passed'] and a['primary_passed']
media=e/'media/joy-current-head-square-cleanup-failed-b08c24e9';m=read(media/'manifest.json');review=read(media/'visual-review.json');assert review['passed'] and review['all_3_panels_inspected'] and review['manifest_sha256']==digest(media/'manifest.json')
d.mkdir(exist_ok=False);paths=set();omitted=[]
for name in ['joy-current-head-square-aa1',run]:
 v=read(e/'checkpoints'/(name+'-local-verification.json'));assert v['all_member_hashes_verified'] and v['all_members_extracted']
 for area,suffix in [('plans','.json'),('preflight','-admission.json'),('benchmarks','-analysis.json'),('checkpoints','-archive-manifest.json'),('checkpoints','-local-verification.json')]:paths.add(e/area/(name+suffix))
 archive=read(e/'checkpoints'/(name+'-archive-manifest.json'));omitted.append(dict(path='evidence/checkpoints/'+archive['archive'],bytes=archive['bytes'],sha256=archive['sha256'],reason='Complete original archive retained locally; member manifest and raw numerical data exported here.'))
 for area in ['block-runs','benchmarks','qualification','reproduction']:
  for item in (e/area).glob('*'+name+'*'):
   for p in ([item] if item.is_file() else item.rglob('*')):
    if p.is_file():paths.add(p)
 for script in read(e/'plans'/(name+'.json'))['sha256']:
  if script.startswith('scripts/'):paths.add(r/script)
for name in ['joy-current-head-square-primary1-failure-review1.json','joy-current-head-square-primary1-cleanup-source-diff1.patch','joy-current-head-lifetime-probe1-audit.json','joy-current-head-tests1-admission.json']:
 paths.add(e/'preflight'/name)
for area in ['diagnostics','reproduction']:
 paths.update(p for p in (e/area/'joy-current-head-lifetime-probe1').rglob('*') if p.is_file())
for name in ['probe_joy_current_head_lifetime1.py','audit_joy_current_head_lifetime1.py','render_joy_current_head_failure_media1.py','package_joy_current_head_failure1.py']:
 paths.add(r/'scripts'/name)
paths.add(e/'plans/joy-current-head-lifetime-probe1.json')
paths.add(r/'inputs/joy-workloads.json');paths.add(r/'inputs/reference-teapot.png')
for name in ['hardware.txt','container-logs/sglang-pip-freeze-final.txt','container-logs/omni-pip-freeze-final.txt']:
 paths.add(e/'preflight'/name)
selected_images={r/source['path'] for row in m['rows'] for source in row['sources']}
paths.update(selected_images)
for path in sorted(paths):
 name=str(path.relative_to(r));target=d/name;sha=digest(path)
 if path.name=='response.json' or (path.suffix=='.png' and path not in selected_images and path!=r/'inputs/reference-teapot.png'):
  omitted.append(dict(path=name,bytes=path.stat().st_size,sha256=sha,reason='Full request/response and media remain in verified raw archive; actual representative source PNGs and all numerical records exported.'));continue
 if path.suffix in ['.pt','.bundle'] or '__pycache__' in path.parts:continue
 target.parent.mkdir(parents=True,exist_ok=True)
 if path.suffix=='.log' and path.stat().st_size>256*1024:
  target=target.with_name(target.name+'.gz');target.write_bytes(gzip.compress(path.read_bytes(),compresslevel=6,mtime=0))
 else:shutil.copy2(path,target)
for p in media.iterdir():
 if p.is_file():target=d/'media'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
lines=['# Joy PR47022: current-head square cleanup revalidation failed','',
 'This report concerns official head b08c24e9630c63790c9f86b48d6c7fc10a0c4301. It preserves the failed frozen cleanup gate. It does not qualify this head, replace the original tested-head report, claim full shape/quality coverage, or add an optimization PR.', '',
 'Frozen SGLang main: af21e2b989e482cfdce39689126562f2209d53ac. Original tested head: 4c135ea92d6b4cbb0ead69bb39976f7dbe995455. Native vLLM-Omni: 817d4c17feb702289a2d2b2cbacfd594c262b7a7, regional compile and CUDNN_ATTN retained.', '',
 'Same physical RTX PRO 6000 Blackwell Server GPU0 (GPU-1565b45e-a702-2015-9776-61f68c36fd59), GPU1 idle; BF16 DiT/encoder/VAE, native backend, high quality, 1024x1024, 30 steps, CFG4, seed1101, batch/concurrency1. Checkpoint 4b41fb25d961f37668750178ccbb380da326201c. Four qualification requests, two excluded warmups and four measured short requests per process. All fresh SG primary arms use SYNC0 without observer/profiler; historical SYNC1 numbers are not pooled.', '',
 'Fixed primary process order: main-original-current-Omni-original-current-current-original-Omni-current-original-main. Main/native Omni have two process replicates each; original/current have four each. All 12 processes and 120 output PNGs completed and remain in the verified raw archive. Separate preceding AA has two main processes and 20 PNGs.', '',
 '| Arm | Measured requests | Mean E2E (s) | p50 (s) | p95 (s) | Peak NVML MiB |','|---|---:|---:|---:|---:|---:|']
for arm,values in a['arms'].items():lines.append(f'| {arm} | {values["n"]} | {values["mean_seconds"]:.9f} | {values["p50_seconds"]:.9f} | {values["p95_seconds"]:.9f} | {values["peak_gpu_memory_bytes"]/2**20:.0f} |')
lines+=['',f'Current remains {a["final_vs_main_latency_reduction_percent"]:.6f}% lower-latency than frozen main and native Omni/current is {a["omni_over_current_speedup"]:.9f}x. Both main/current and Omni/current pairs improve. These positive comparator results do not satisfy the separate cleanup gate.', '',
 f'Cleanup mean regression: {a["cleanup_mean_latency_change_percent"]:.9f}% vs fresh AA noise {a["AAmax_noise_percent"]:.9f}%. Four-process-pair log-t95 upper regression: {a["cleanup_t95_regression_upper_percent"]:.9f}%, above the frozen <0.1% limit. Both cleanup checks fail. Every sample and original threshold is retained.', '',
 'Every current-head process samples a peak 138 MiB above the original tested head. A separate unprofiled actual SM120 production-MLP probe confirms current retains its temporary normalized input through fc_out, while original releases it before fc_out: allocator differences are exactly 56/64 MiB for 7168/8192 BF16 tokens. All eight probe outputs agree, and eight separate traces have matching two-kernel dispatches per shape. This proves input-lifetime memory impact; it does not attribute the entire 138 MiB NVML difference or measured E2E difference to that one cause.', '',
 'Actual current/main high-quality square outputs pass the frozen numerical image checks. The three displayed pairs come from fresh primary qualification requests, with complete 1:1 image pixels and source identities in media/manifest.json. Fresh wide/tall, quality-switching and synchronized denoise/profile supplement for this official head were not run after this failure; no new BCG or full-model LoRA/compile result is claimed. A separately identified minimal repair is being tested and is not evidence that official b08c passed.', '',
 'Reproduction: install the recorded source commits and original checkpoint, preserve inputs and native launchers, then run run_joy_current_head_blocks1.py with joy-current-head-square-aa1.json. Archive/verify/analyze/admit AA before the primary plan. Use a fresh output directory, retaining all fixed process pairs and requests; do not overwrite these runs. SYNC0 stage perf dumps are retained but do not establish denoise-stage latency.', '',
 'Complete raw archives remain local with public per-member SHA256 manifests. Raw numerical records, source/ownership metadata, telemetry, launchers, plans, admissions, relevant source diff and independent lifetime diagnostic are exported. Unselected full PNGs and repeated response JSON are omitted here with original hashes recorded; representative actual source PNGs remain included. No checkpoint weights or transfer bundles are published. All artifacts concern failed/partial validation, not a successful final qualification.', '']
(d/'README.md').write_text('\n'.join(lines))
(d/'omitted-content.json').write_text(json.dumps(omitted,indent=2)+'\n')
files={str(p.relative_to(d)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in d.rglob('*') if p.is_file()};assert all(v['bytes']<100*1024**2 for v in files.values())
(d/'evidence-files-sha256.json').write_text(json.dumps(files,indent=2)+'\n');print(json.dumps(dict(files=len(files),bytes=sum(x['bytes'] for x in files.values()),panels=3,cleanup_passed=False,publication_pending=True)))
