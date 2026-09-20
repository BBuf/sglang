"""Audit all native requests, fixed comparisons, NCU and saved image pixels."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

root = Path('/campaign')
art = root / 'artifacts/flux2-klein-base-4b'
for name in ['measure-klein-qk-fullwarm', 'validate-klein-quality', 'analyze-klein-final-profile']:
    done = root / f'{name}.exit'
    while not done.exists():
        time.sleep(5)
    assert done.read_text().strip() == '0', name
base_sha = '80da4432d085ed4d6166ef643d9fd2b829dbb0c5'
candidate_sha = 'a97bdd428b27aa550b491e7f02256af336e7ab37'
profile_sha = '454ee669a40c4119967a0909449237dfbdc63998'
expected_sha = '8c402631ec1e5956efe64e38e4de31b18a32f12e2d56dda4612703a7e739a9d0'
assert subprocess.check_output(['git', '-C', str(root/'candidate-klein-qk-final'), 'diff', profile_sha, candidate_sha, '--', 'python', 'test'], text=True) == ''

def group_summary(rows):
    groups = {}
    for round in sorted({r['round'] for r in rows}):
        selected = [r for r in rows if r['round'] == round]
        assert [r['arm'] for r in selected] == ['baseline', 'candidate', 'candidate', 'baseline']
        means = {}
        for key in ['worker_s', 'denoise_s', 'client_saved_s']:
            a, b = [statistics.mean(r[key] for r in selected if r['arm'] == arm) for arm in ['baseline', 'candidate']]
            means[key] = dict(baseline=a, candidate=b, reduction_pct=(1-b/a)*100)
        groups[str(round)] = dict(rows=selected, means=means,
            qualifies=all(means[k]['reduction_pct'] >= 1.5 for k in ['worker_s', 'client_saved_s']))
    return groups

comparisons = {}
for version in ['v2', 'v3', 'fullwarm']:
    rows = json.loads((art/f'qk-abba-{version}.json').read_text())
    assert len(rows) == 8, (version,len(rows))
    for row in rows:
        r = json.loads((art/row['label']/'result.json').read_text())
        assert not r['error'] and r['quality']=='lossless' and not r['breakable_cuda_graph']
        assert r['output_sha256'] == [expected_sha]
        if version == 'fullwarm':
            assert (art/row['label']/'source.txt').read_text().splitlines()[0] == (base_sha if row['arm']=='baseline' else candidate_sha)
            preset = json.loads((art/row['label']/'preset.json').read_text())
            assert '--warmup-steps=50' in preset['extra_args']
    comparisons[version] = group_summary(rows)

reference_file = next((art/'baseline-a1').glob('*.png'))
reference = np.asarray(Image.open(reference_file).convert('RGB'))
assert reference.shape == (1024,1024,3)
outputs = []
for file in sorted(art.glob('*/result.json')):
    result = json.loads(file.read_text())
    pngs = list(file.parent.glob('*.png'))
    assert len(pngs)==1, file.parent.name
    pixels = np.asarray(Image.open(pngs[0]).convert('RGB'))
    assert pixels.shape==reference.shape
    sha = hashlib.sha256(pngs[0].read_bytes()).hexdigest()
    assert result['output_sha256']==[sha]
    exact = bool(np.array_equal(pixels,reference))
    ssim = 1.0 if exact else float(structural_similarity(reference,pixels,channel_axis=-1,data_range=255))
    psnr = None if exact else float(peak_signal_noise_ratio(reference,pixels,data_range=255))
    if result['quality']=='lossless':
        assert exact and sha==expected_sha,file.parent.name
    if result['breakable_cuda_graph']:
        assert result['error'] and not result['bcg_capture_detected'] and '[diffusion bcg] disabled' in result['bcg_invalid_signals']
    else:
        assert not result['error'],file.parent.name
    outputs.append(dict(label=file.parent.name,quality=result['quality'],valid_request=not result['error'],
        pixel_exact=exact,sha256=sha,ssim=ssim,psnr_db=psnr,peak_memory_gib=result.get('peak_memory_gb'),
        source=(file.parent/'source.txt').read_text().splitlines()[0]))
high=[o for o in outputs if o['label'] in ['final-baseline-high','final-candidate-high']]
assert len(high)==2 and high[0]['sha256']==high[1]['sha256']
assert all(o['ssim']>=0.95 and (o['psnr_db'] is None or o['psnr_db']>=28) for o in high)

profiles={}
for arm,label in [('baseline','baseline-profile'),('candidate','qk-v3-candidate-profile')]:
    forward=json.loads((art/label/'traces/forward3-evidence.json').read_text())
    chain=json.loads((art/label/'traces/qk-chain-evidence.json').read_text())
    assert chain['scopes']==20 and chain['kernels']==(120 if arm=='baseline' else 20)
    profiles[arm]=dict(model_calls=forward['model_calls'],kernel_count=forward['kernel_count'],
        gpu_window_ms=forward['gpu_window_ms'],gpu_union_ms=forward['gpu_union_ms'],qk_chain=chain)

ncu_old=json.loads((root/'profile/klein-qk-rope-v2-h200/key-metrics.json').read_text())
ncu_new=json.loads((root/'profile/klein-qk-rope-v3-h200/key-metrics.json').read_text())
report=dict(qualified=all(g['qualifies'] for g in comparisons['fullwarm'].values()),
    checkpoint='black-forest-labs/FLUX.2-klein-base-4B',checkpoint_revision='a3b4f4849157f664bdbc776fd7453c2783562f4d',
    baseline=base_sha,candidate=candidate_sha,profile_candidate=profile_sha,
    profile_candidate_difference='Documentation only; python/test trees are identical.',
    comparisons=comparisons,outputs=outputs,high_baseline_candidate_byte_exact=True,profiles=profiles,
    microbench=json.loads((art/'qk-microbench-v3.json').read_text()),
    ncu=dict(baseline=ncu_old['baseline'],gather_prototype=ncu_old['candidate'],pair_candidate=ncu_new['candidate']),
    timing_notes='v2/v3 groups use native default one-step same-shape request warmup and include all slow observations. The separately fixed fullwarm groups use fifty-step same-shape request warmup on both arms. Loading/warmup excluded from request timer; no profile timing used as E2E.',
    triage_correction='FA3 is attention. RMSNorm/pointwise kernels are not GEMMs. This dense BF16 model has no MoE route. Existing SwiGLU, adaLN and residual fusions already run.',
)
(art/'final-evidence.json').write_text(json.dumps(report,indent=2))

canvas=Image.new('RGB',(1040,1112),'white');draw=ImageDraw.Draw(canvas)
for index,(label,title) in enumerate([
 ('qk-fullwarm-r2-baseline-a1','Lossless baseline'),('qk-fullwarm-r2-candidate-b1','Lossless fused'),
 ('final-baseline-high','High baseline'),('final-candidate-high','High fused')]):
    x=(index%2)*520;y=(index//2)*556
    draw.text((x+8,y+8),title,fill='black')
    picture=Image.open(next((art/label).glob('*.png'))).convert('RGB')
    picture.thumbnail((512,512))
    canvas.paste(picture,(x+4,y+32))
out=art/'output-comparison';out.mkdir(exist_ok=True)
canvas.save(out/'image-comparison.png')

import torch
environment=dict(torch=torch.__version__,torch_git=torch.version.git_version,cuda=torch.version.cuda,
    versions={key:importlib.metadata.version(key) for key in ['triton','transformers','diffusers','sglang-kernel']},
    gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True),
    used_gpu_indices=[0],tp=1,sp=1,quality='lossless',dtype='bfloat16',mode='eager',
    residency='DiT, text encoder, VAE resident; manual performance mode; no torch.compile',
    request=dict(width=1024,height=1024,num_frames=1,steps=50,cfg=4,seed=42,
        prompt='A futuristic cyberpunk city at night, neon lights reflecting on wet streets'),
    headline_warmup='same-shape native request warmup, warmup_steps=50 in both arms')
(art/'environment.json').write_text(json.dumps(environment,indent=2))
print(json.dumps(dict(qualified=report['qualified'],groups={k:{r:g['means'] for r,g in v.items()} for k,v in comparisons.items()},outputs=len(outputs),high=high),indent=2),flush=True)
