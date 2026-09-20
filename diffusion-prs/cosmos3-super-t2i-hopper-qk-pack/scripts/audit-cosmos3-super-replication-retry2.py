"""Audit fixed native Super T2I groups, launch-owned profiles and every saved image."""
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

root = Path('/campaign')
art = root / 'artifacts/cosmos3-super-t2i'
done = root / 'validate-cosmos3-super-replication-retry2.exit'
while not done.exists(): time.sleep(5)
assert done.read_text().strip() == '0'
baseline = '80da4432d085ed4d6166ef643d9fd2b829dbb0c5'
candidate = '1a7e397700ad6f6d7c1bb5af719411c9937bec1f'
groups = {}
for repeat in (1, 2, 3):
    rows = []
    for arm in ('a1', 'b1', 'b2', 'a2'):
        label = f'final-r{repeat}-{arm}'
        result = json.loads((art / label / 'result.json').read_text())
        assert not result['error'] and result['quality'] == 'lossless'
        assert not result['breakable_cuda_graph']
        source = (art / label / 'source.txt').read_text().splitlines()[0]
        assert source == (baseline if arm.startswith('a') else candidate)
        log = re.sub(r'\x1b\[[0-9;]*m', '', (root / f'cosmos3-super-t2i-{label}.log').read_text())
        client = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds', log)
        assert len(client) == 1, label
        rows.append(dict(label=label, arm=arm, source=source, worker_s=result['e2e_latency_s'],
                         client_s=float(client[0]), denoise_s=result['denoise_latency_s'],
                         peak_gib=result['peak_memory_gb']))
    means = {}
    for key in ('worker_s', 'client_s', 'denoise_s'):
        a = statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
        b = statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
        means[key] = dict(baseline=a, candidate=b, reduction_pct=(a-b)/a*100)
    groups[f'r{repeat}'] = dict(rows=rows, means=means,
        qualified=all(means[k]['reduction_pct'] >= 1.5 for k in ('worker_s', 'client_s')))
assert all(g['qualified'] for g in groups.values()), groups

profiles = {}
for arm in ('baseline', 'candidate'):
    label = f'final-{arm}-profile'
    result = json.loads((art / label / 'result.json').read_text())
    assert not result['error'] and len(result['native_profile_traces']) == 1
    source = Path(result['native_profile_traces'][0]['path'])
    with (art / label / 'inspect.txt').open('w') as out:
        subprocess.run(['python', str(root / 'inspect-cosmos3-super-trace.py'), str(source)], stdout=out, check=True)
    with gzip.open(source.parent / 'forward3.trace.json.gz', 'rt') as stream:
        events = json.load(stream)['traceEvents']
    launches = sorted((e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
        and 'correlation' in e.get('args', {}) and 'LaunchKernel' in e.get('name', '')), key=lambda e:e['ts'])
    times = [e['ts'] for e in launches]
    kernels = collections.defaultdict(list)
    for event in events:
        if event.get('cat') == 'kernel': kernels[event['args']['correlation']].append(event)
    def owned(scope):
        lo=bisect.bisect_left(times,scope['ts']); hi=bisect.bisect_left(times,scope['ts']+scope['dur'])
        return [k for launch in launches[lo:hi] for k in kernels[launch['args']['correlation']]]
    qk = [e for e in events if e.get('cat') == 'python_function' and e.get('name','').endswith(
        ': _apply_qwen3_qk_norm_rope_split' if arm == 'baseline' else ': _apply_qwen3_qk_norm_rope_pack_kv')]
    cats = [e for e in events if e.get('name') == 'aten::cat' and e.get('args',{}).get('Input Dims') ==
        [[[1,23,4,128],[1,1024,4,128]], []]]
    assert len(qk) == 64 and len(cats) == (128 if arm == 'baseline' else 0), (arm,len(qk),len(cats))
    selected = [k for scope in qk+cats for k in owned(scope)]
    assert len({(k['ts'],k['name'],k['args']['correlation']) for k in selected}) == len(selected)
    summary = collections.defaultdict(lambda:[0,0.0])
    for kernel in selected:
        summary[kernel['name']][0] += 1; summary[kernel['name']][1] += kernel['dur']/1000
    attribution = dict(qk_scopes=len(qk), prefix_cat_scopes=len(cats), kernels=len(selected),
        gpu_ms=sum(k['dur'] for k in selected)/1000,
        by_kernel=[dict(name=name,count=row[0],gpu_ms=row[1]) for name,row in sorted(summary.items(),key=lambda kv:-kv[1][1])],
        note='Disjoint QKNorm/RoPE scopes plus native UND-prefix cats, matched by input shape. CUDA launch correlation selects owned kernels. Rank 0 only; diagnostic cumulative device time, not request latency.')
    (source.parent/'qk-pack-attribution.json').write_text(json.dumps(attribution,indent=2))
    with (art/label/'triage.txt').open('w') as out:
        subprocess.run(['python',str(root/'llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py'),
            '--framework','sglang','--input',str(source.parent/'forward3.trace.json.gz')],stdout=out,check=True)
    profiles[arm] = json.loads((source.parent/'forward3-evidence.json').read_text()) | {'qk_pack':attribution}

reference_path = next((art/'final-r2-a1').glob('*.png'))
reference = np.asarray(Image.open(reference_path).convert('RGB'))
assert reference.shape == (1024,1024,3)
outputs = []
for path in sorted(art.glob('*/result.json')):
    result = json.loads(path.read_text()); valid = not result['error']; label = path.parent.name
    artifacts = result.get('output_artifacts') or list(path.parent.glob('*.png'))
    if result.get('breakable_cuda_graph', False):
        assert not valid and '[diffusion bcg] disabled' in result['bcg_invalid_signals'], label
    if not artifacts:
        assert not valid
        outputs.append(dict(label=label,valid_request=False,artifact_produced=False,error_reason=result.get('error_reason')))
        continue
    assert len(artifacts) == 1
    image_path = Path(artifacts[0]); digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if result.get('output_sha256'): assert result['output_sha256'] == [digest]
    pixels = np.asarray(Image.open(image_path).convert('RGB')); assert pixels.shape == reference.shape
    exact = np.array_equal(pixels,reference)
    ssim = 1.0 if exact else float(structural_similarity(reference,pixels,channel_axis=-1,data_range=255))
    psnr = None if exact else float(peak_signal_noise_ratio(reference,pixels,data_range=255))
    quality_pass = exact if result['quality']=='lossless' else (ssim>=.95 and (psnr is None or psnr>=28))
    assert quality_pass, (label,ssim,psnr)
    outputs.append(dict(label=label,quality=result['quality'],valid_request=valid,path=str(image_path),
        sha256=digest,pixel_exact=exact,ssim=ssim,psnr_db=psnr,quality_pass=quality_pass,
        bcg_capture_detected=result.get('bcg_capture_detected'),bcg_invalid_signals=result.get('bcg_invalid_signals',[])))
high=[row for row in outputs if row['label'] in ('final-baseline-high-eager','final-candidate-high-eager')]
assert len(high)==2
high_exact=high[0]['sha256']==high[1]['sha256']
comparison=art/'output-comparison'; comparison.mkdir(exist_ok=True)
figure=Image.new('RGB',(2048,2112),'white'); draw=ImageDraw.Draw(figure)
for index,label in enumerate(('final-r2-a1','final-r2-b1','final-baseline-high-eager','final-candidate-high-eager')):
    picture=Image.open(next((art/label).glob('*.png'))).convert('RGB')
    x=(index%2)*1024; y=(index//2)*1056
    figure.paste(picture,(x,y+32));draw.text((x+12,y+10),label,fill='black')
figure.save(comparison/'image-comparison.png')
report=dict(qualified=True,baseline=baseline,candidate=candidate,groups=groups,profiles=profiles,
    outputs=outputs,high_baseline_candidate_byte_exact=high_exact,
    note='Three fixed ABBA groups on pinned sources. Group 3 was added because group 2 had baseline worker/client latency outliers; no original rows were dropped. All native saved-output requests retained; failed offline admission and disabled BCG rows are not performance evidence. Profile and kernel CUDA Graph measurements do not claim model BCG support.')
(art/'initial-two-group-evidence.json').write_bytes((art/'final-evidence.json').read_bytes())
(art/'final-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(groups=groups,outputs=len(outputs),high_baseline_candidate_byte_exact=high_exact,
    profile_attribution={arm:report['qk_pack'] for arm,report in profiles.items()}),indent=2))
