"""Audit the shared LongCat T2I fast path and delete its owned checkpoint."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import numpy as np
from PIL import Image

root = Path('/campaign')
art = root/'artifacts/longcat-image'
done = root/'advance-longcat-image.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
rows = []
means = {}
for mode in ('eager', 'bcg'):
    for arm in ('a1', 'b1', 'b2', 'a2'):
        label = f'gelu-cat-{mode}-r1-{arm}'
        assert (root/f'longcat-image-{label}.exit').read_text().strip() == '0'
        result = json.loads((art/label/'result.json').read_text())
        assert not result['error']
        assert result['bcg_capture_detected'] == (mode == 'bcg')
        log = re.sub(r'\x1b\[[0-9;]*m', '', (root/f'longcat-image-{label}.log').read_text())
        client = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds', log)
        assert len(client) == 1
        rows.append(dict(label=label, mode=mode, arm=arm,
            worker_s=result['e2e_latency_s'], client_s=float(client[0]),
            denoise_s=result['denoise_latency_s'], peak_gib=result['peak_memory_gb']))
    means[mode] = {}
    for key in ('worker_s', 'client_s', 'denoise_s'):
        a = statistics.mean(r[key] for r in rows if r['mode'] == mode and r['arm'].startswith('a'))
        b = statistics.mean(r[key] for r in rows if r['mode'] == mode and r['arm'].startswith('b'))
        means[mode][key] = dict(baseline=a, candidate=b, reduction_pct=(a-b)/a*100)
profiles = {}
for label in ('baseline-profile', 'candidate-profile', 'candidate-bcg-profile'):
    assert (root/f'longcat-image-{label}.exit').read_text().strip() == '0'
    result = json.loads((art/label/'result.json').read_text())
    assert not result['error'] and result['native_profile_traces']
    source = next((art/label/'traces').glob('*-3_steps-*.trace.json.gz'))
    with gzip.open(source, 'rt') as f:
        events = json.load(f)['traceEvents']
    models = sorted((e for e in events if e.get('name') == 'nn.Module: LongCatImageTransformer2DModel_0'), key=lambda e:e['ts'])
    graphs = sorted((e for e in events if e.get('name') == 'cudaGraphLaunch'), key=lambda e:e['ts'])
    if models:
        assert len(models) == 8, len(models)
        lo, hi = models[2]['ts'], models[4]['ts']
    else:
        assert label == 'candidate-bcg-profile' and len(graphs) == 50*2*31, len(graphs)
        lo, hi = graphs[2*31]['ts'], graphs[4*31]['ts']
    assert not any(e.get('cat') == 'kernel' and any(e['ts'] < t < e['ts']+e['dur'] for t in (lo, hi)) for e in events)
    kernels = [e for e in events if e.get('cat') == 'kernel' and lo <= e['ts'] < hi]
    fused = [e for e in kernels if 'sglang::gelu_tanh_cat_kernel' in e['name']]
    if label != 'baseline-profile':
        assert len(fused) == 40, len(fused)
    profiles[label] = dict(path=str(source), model_calls=len(models), graph_launches=len(graphs),
        window_ms=(hi-lo)/1000, crossing_kernels=0, kernel_count=len(kernels),
        fused_calls=len(fused), fused_gpu_ms=sum(e['dur'] for e in fused)/1000)
reference = np.asarray(Image.open(next((art/'gelu-cat-eager-r1-a1').glob('*.png'))).convert('RGB'))
images = []
for p in sorted(art.glob('*/*.png')):
    assert np.array_equal(reference, np.asarray(Image.open(p).convert('RGB'))), p
    images.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), pixel_exact=True))
assert len(images) == 11 and len({r['sha256'] for r in images}) == 1
final = dict(additional_pr_qualified=False, existing_pr=40384, rows=rows, means=means,
    profiles=profiles, images=images,
    note='Shared-kernel family validation; both BCG arms replay real graphs. One ABBA per mode, no additional PR counted.')
(art/'final-evidence.json').write_text(json.dumps(final, indent=2))
lock = (root/'gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
selected = {line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
for _ in range(30):
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], text=True)
    if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()):
        break
    time.sleep(1)
else:
    raise RuntimeError('Assigned GPU remains busy; no process was killed')
helper = root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec = importlib.util.spec_from_file_location('campaign_bench', helper)
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
record = bench._cleanup_model_cache(root/'model-caches', root/'model-caches/longcat-image-cycle',
    root/'artifacts/cache-cleanup.jsonl', 'longcat-image', 'cycle',
    'Completed shared-kernel eager/BCG ABBA and native profiles; all eleven PNGs exact')
assert record['after'] == dict(file_count=0, weight_file_count=0, total_bytes=0)
assert not (root/'model-caches/longcat-image-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record, indent=2))
print(json.dumps(dict(means=means, profiles=profiles, cleanup=record), indent=2))
