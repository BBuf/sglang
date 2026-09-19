"""Audit ordinary Edit regression evidence and remove its owned checkpoint."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess

from PIL import Image
import numpy as np

root = Path('/campaign')
art = root/'artifacts/longcat-image-edit'
assert (root/'resume-longcat-edit.exit').read_text().strip() == '0'
labels = ['gelu-cat-r1-a1','gelu-cat-r1-b1','gelu-cat-r1-b2-retry1','gelu-cat-r1-a2']
rows = []
for label, arm in zip(labels, ['a1','b1','b2','a2']):
    assert (root/f'longcat-edit-{label}.exit').read_text().strip() == '0'
    result = json.loads((art/label/'result.json').read_text())
    log = re.sub(r'\x1b\[[0-9;]*m','',(root/f'longcat-edit-{label}.log').read_text())
    client = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log)
    assert len(client) == 1
    rows.append(dict(label=label,arm=arm,worker_s=result['e2e_latency_s'],
        client_s=float(client[0]),denoise_s=result['denoise_latency_s'],
        sha256=result['output_sha256'][0],peak_gib=result['peak_memory_gb']))
means = {}
for key in ('worker_s','client_s','denoise_s'):
    a = statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
    b = statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
    means[key] = dict(baseline=a,candidate=b,reduction_pct=(a-b)/a*100)
profile = {}
for name in ('baseline','candidate'):
    assert (root/f'longcat-edit-{name}-profile.exit').read_text().strip() == '0'
    source = next((art/f'{name}-profile/traces').glob('*-3_steps-*.trace.json.gz'))
    with gzip.open(source,'rt') as f:events=json.load(f)['traceEvents']
    models=sorted((e for e in events if e.get('name')=='nn.Module: LongCatImageTransformer2DModel_0'),key=lambda e:e['ts'])
    assert len(models)==8
    lo,hi=models[2]['ts'],models[4]['ts']
    assert not any(e.get('cat')=='kernel' and any(e['ts']<t<e['ts']+e['dur'] for t in (lo,hi)) for e in events)
    selected=[e for e in events if e.get('cat')=='kernel' and lo<=e['ts']<hi]
    if name=='baseline':
        ops=[e for e in events if lo<=e.get('ts',-1)<hi and (
             e.get('name')=='aten::gelu' and e.get('args',{}).get('Input Dims')==[[1,9233,12288],[]] or
             e.get('name')=='aten::cat' and e.get('args',{}).get('Input Dims')==[[[1,9233,3072],[1,9233,12288]],[]])]
        ids={e['args']['External id'] for e in ops}
        kernels=[e for e in selected if e.get('args',{}).get('External id') in ids]
        assert len(kernels)==80
    else:
        kernels=[e for e in selected if 'sglang::gelu_tanh_cat_kernel' in e['name']]
        assert len(kernels)==40
    profile[name]=dict(path=str(source),model_calls=8,crossing_kernels=0,
        window_ms=(hi-lo)/1000,kernel_count=len(selected),
        gelu_cat_kernels=len(kernels),gelu_cat_gpu_ms=sum(e['dur'] for e in kernels)/1000)
reference=np.asarray(Image.open(next((art/labels[0]).glob('*.png'))).convert('RGB'))
images=[]
for p in sorted(art.glob('*/*.png')):
    array=np.asarray(Image.open(p).convert('RGB'))
    assert np.array_equal(reference,array),p
    images.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pixel_exact=True))
assert len(images)==7 and len({r['sha256'] for r in images})==1
bcg=json.loads((art/'bcg-applicability/result.json').read_text())
assert bcg['error'] and '[diffusion bcg] disabled' in bcg['bcg_invalid_signals']
repo=root/'candidate-longcat-gelu-cat'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='10410a20b6d2716c1489a3132dc79c775dc297bb'
final=dict(additional_pr_qualified=False,existing_pr=40384,rows=rows,means=means,profile=profile,images=images,bcg=bcg,
    note='Shared kernel regression validation. Worker mean improves1.64%, but candidate saved-client outlier19.22s prevents an eager saved-request E2E claim. Initial B2 was rejected before admission for lingering CUDA context; the unique retry is retained. No duplicate PR is counted.')
(art/'final-evidence.json').write_text(json.dumps(final,indent=2))
lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
assert not any(line.split(',')[0].strip() in selected for line in processes.splitlines()),processes
helper=root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
record=bench._cleanup_model_cache(root/'model-caches',root/'model-caches/longcat-image-edit-cycle',
    root/'artifacts/cache-cleanup.jsonl','longcat-image-edit','cycle',
    'Completed shared-kernel native regression ABBA and paired CFG profiles; no additional saved-request performance claim; all images exact')
assert record['after']==dict(file_count=0,weight_file_count=0,total_bytes=0)
assert not (root/'model-caches/longcat-image-edit-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(means=means,profile=profile,cleanup=record),indent=2))
