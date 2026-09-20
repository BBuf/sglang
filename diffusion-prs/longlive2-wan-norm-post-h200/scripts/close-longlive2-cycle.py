"""Qualify the independent I2V workload, preserve T2V limitations, and clean owned weights."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import time
root=Path('/campaign');art=root/'artifacts/longlive2'
for name in ['longlive2-norm-audit-retry3','diagnose-longlive2-client-retry2']:
    done=root/f'{name}.exit'
    while not done.exists():time.sleep(5)
    assert done.read_text().strip()=='0',done
report=json.loads((art/'final-evidence.json').read_text())
assert report['candidate']=='be05e7e6c24252db11458621f2d06f3969be08d1'
assert all(report['groups'][f'i2v-r{r}']['qualified'] for r in range(1,5))
assert not report['groups']['t2v-r4']['qualified']
assert len(report['outputs'])==56
assert sum(r['valid_request'] for r in report['outputs'])==48
assert all(r.get('quality_pass',True) for r in report['outputs'] if r['valid_request'])
assert all(r['byte_exact'] for r in report['outputs'] if r['valid_request'] and r['quality']=='lossless')
qualification=dict(qualified=True,qualified_workloads=['longlive2-i2v'],unqualified_workloads=['longlive2-t2v'],
    candidate=report['candidate'],basis='All four independent I2V ABBA groups exceed 1.5% in both saved-request worker and client means, including final-source r3/r4. No claim that both model workloads qualify.',
    worker_improvement_pct={k:v['means']['worker_s']['reduction_pct'] for k,v in report['groups'].items()},
    client_improvement_pct={k:v['means']['client_s']['reduction_pct'] for k,v in report['groups'].items()},
    limitations=['T2V r1B1 and r4B2 have post-save client outliers; retain all samples and make no T2V client E2E speed claim.',
    'I2V r4A1 baseline client 7.08s inflates r4 mean improvement. Quote r3 client reduction 5.27%, not r4 12.54%, as representative.',
    'Native I2V CLI requests 960x928 but preserves source aspect ratio and produces 1152x768. All 24 baseline/candidate videos share this geometry. The preset emits a supported-resolution advisory.',
    'Actual native BCG is disabled in both LongLive 2 configurations. Standalone CUDA graph replay tests do not establish native BCG support.',
    'Final native peak memory increases from 59.642578125 to 61.01953125 GiB. Performance mode auto resolves all components resident in both arms.',
    'Diagnostic py-spy/GC runs and all torch profiles are excluded from performance claims.'])
(art/'qualification-review.json').write_text(json.dumps(qualification,indent=2))
lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
for _ in range(30):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()):break
    time.sleep(1)
else:raise RuntimeError('Assigned GPU busy; no process was killed')
helper=root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
record=bench._cleanup_model_cache(root/'model-caches',root/'model-caches/longlive2-cycle',root/'artifacts/cache-cleanup.jsonl','longlive2','cycle',
    'Final source tests, paired full profiles, repeated independent I2V qualification, T2V limitations, client diagnostics and decoded media audit complete')
assert record['after']==dict(file_count=0,weight_file_count=0,total_bytes=0)
assert not (root/'model-caches/longlive2-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(qualification=qualification,cleanup=record),indent=2))
