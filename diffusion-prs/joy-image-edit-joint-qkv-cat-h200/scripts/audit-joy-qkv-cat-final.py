"""Audit the final Joy native requests, profiles, hashes and fixed ABBA groups."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

root = Path('/campaign')
art = root / 'artifacts/joy-image-edit'
done = root / 'validate-joy-qkv-cat-final.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
base_sha = '80da4432d085ed4d6166ef643d9fd2b829dbb0c5'
candidate_sha = '3b00154ddb33ac436906a493c9706ed3855ec918'
v2_sha = '0cff694f701987ba694954a4cf54257f7c3881e6'
expected_sha = '76cfd40eb83feacb1b0292e171b9beb2a8cd7eb7fadc606cb1632e886d960ce6'
comparisons = {}
for version, head in [('v2', v2_sha), ('final', candidate_sha)]:
    rows = json.loads((art / f'qkv-abba-fullwarm-{version}.json').read_text())
    assert len(rows) == 8
    groups = {}
    for round_index in [1, 2]:
        selected = [r for r in rows if r['round'] == round_index]
        assert [r['arm'] for r in selected] == ['baseline', 'candidate', 'candidate', 'baseline']
        for row in selected:
            folder = art / row['label']
            native = json.loads((folder / 'result.json').read_text())
            assert not native['error'] and not native['breakable_cuda_graph']
            assert native['quality'] == 'lossless' and native['output_sha256'] == [expected_sha]
            assert (folder / 'source.txt').read_text().splitlines()[0] == (base_sha if row['arm'] == 'baseline' else head)
            assert '--warmup-steps=40' in json.loads((folder / 'preset.json').read_text())['extra_args']
        means = {}
        for key in ['worker_s', 'denoise_s', 'client_saved_s']:
            baseline, candidate = [statistics.mean(r[key] for r in selected if r['arm'] == arm) for arm in ['baseline', 'candidate']]
            means[key] = dict(baseline=baseline, candidate=candidate, reduction_pct=(1-candidate/baseline)*100)
        groups[str(round_index)] = dict(rows=selected, means=means,
            qualifies=all(means[k]['reduction_pct'] >= 1.5 for k in ['worker_s', 'client_saved_s']))
    comparisons[version] = groups

reference_file = next((art / 'baseline-retry1').glob('*.png'))
reference = np.asarray(Image.open(reference_file).convert('RGB'))
assert reference.shape == (1024, 1024, 3)
outputs = []
failed_requests = []
for result_file in sorted(art.glob('*/result.json')):
    label = result_file.parent.name
    result = json.loads(result_file.read_text())
    pngs = list(result_file.parent.glob('*.png'))
    if not pngs:
        assert label == 'baseline-a1' and result['error']
        failed_requests.append(dict(label=label, reason='Unsupported --cfg-parallel-degree flag, corrected before admission'))
        continue
    assert len(pngs) == 1
    pixels = np.asarray(Image.open(pngs[0]).convert('RGB'))
    assert pixels.shape == reference.shape
    sha = hashlib.sha256(pngs[0].read_bytes()).hexdigest()
    exact = bool(np.array_equal(pixels, reference))
    ssim = 1.0 if exact else float(structural_similarity(reference, pixels, channel_axis=-1, data_range=255))
    psnr = None if exact else float(peak_signal_noise_ratio(reference, pixels, data_range=255))
    quality = result.get('quality', 'high' if 'high' in label else 'lossless')
    bcg = result.get('breakable_cuda_graph', 'bcg' in label)
    if quality == 'lossless':
        assert exact and sha == expected_sha, label
    if bcg:
        assert result['error'] and not result['bcg_capture_detected']
        assert '[diffusion bcg] disabled' in result['bcg_invalid_signals']
    else:
        assert not result['error'] and result['output_sha256'] == [sha]
    outputs.append(dict(label=label, quality=quality, bcg=bcg, valid_request=not result['error'],
        pixel_exact=exact, sha256=sha, ssim=ssim, psnr_db=psnr,
        peak_memory_gib=result.get('peak_memory_gb'),
        source=(result_file.parent / 'source.txt').read_text().splitlines()[0]))
high = [o for o in outputs if o['label'] in ['qkv-v2-baseline-high', 'qkv-final-candidate-high']]
assert len(high) == 2 and high[0]['sha256'] == high[1]['sha256']
assert all(o['ssim'] >= .95 and (o['psnr_db'] is None or o['psnr_db'] >= 28) for o in high)

profiles = {}
for arm, label in [('baseline', 'baseline-profile-retry1'), ('candidate', 'qkv-final-candidate-profile')]:
    evidence = json.loads((art / label / 'traces/forward3-evidence.json').read_text())
    cats = [k for k in evidence['top_kernels'] if 'CatArrayBatchedCopy' in k['name']]
    fused = [k for k in evidence['top_kernels'] if '_joint_qkv_cat_kernel' in k['name']]
    assert sum(k['count'] for k in cats) == (120 if arm == 'baseline' else 0)
    assert sum(k['count'] for k in fused) == (0 if arm == 'baseline' else 40)
    profiles[arm] = dict(model_calls=evidence['model_calls'], kernel_count=evidence['kernel_count'],
        gpu_window_ms=evidence['gpu_window_ms'], gpu_union_ms=evidence['gpu_union_ms'],
        cats=cats, fused=fused)

report = dict(qualified=all(g['qualifies'] for g in comparisons['final'].values()),
    checkpoint='jdopensource/JoyAI-Image-Edit-Diffusers', checkpoint_revision='4b41fb25d961f37668750178ccbb380da326201c',
    baseline=base_sha, candidate=candidate_sha, ncu_candidate=v2_sha,
    ncu_candidate_difference='Same Triton kernel; final commit defers backend import and updates documentation. Final native tests and profiles measured separately.',
    comparisons=comparisons, outputs=outputs, failed_requests=failed_requests, profiles=profiles,
    high_baseline_candidate_byte_exact=True,
    microbench=json.loads((art / 'qkv-cat-microbench-v2.json').read_text()),
    ncu=json.loads((root / 'profile/joy-qkv-cat-v2-h200/analysis/key-metrics.json').read_text()),
    timing_notes='All fixed ABBA rows retained. Both arms use native same-shape forty-step request warmup. Loading/warmup excluded from request timing. Client time includes saved PNG. Profile timings are never E2E. V2 round 1 has a 15.36 s client baseline outlier.',
    triage_correction='FA3 is attention; CuTe normalization is not GEMM. This dense BF16 model has no MoE route. Text Q/K has normalization but no RoPE for this request.',
    bcg_note='Native pipeline explicitly disables BCG in all tested lossless/high cases. Standalone kernel graph replay is correctness/microbenchmark evidence only.')
(art / 'final-evidence.json').write_text(json.dumps(report, indent=2))

out = art / 'output-comparison'
out.mkdir(exist_ok=True)
canvas = Image.new('RGB', (1040, 1112), 'white')
draw = ImageDraw.Draw(canvas)
for index, (label, title) in enumerate([
    ('qkv-final-fullwarm-r2-baseline-a1', 'Lossless baseline'),
    ('qkv-final-fullwarm-r2-candidate-b1', 'Lossless joint copy'),
    ('qkv-v2-baseline-high', 'High baseline'),
    ('qkv-final-candidate-high', 'High joint copy'),
]):
    x, y = (index % 2)*520, (index // 2)*556
    draw.text((x+8, y+8), title, fill='black')
    picture = Image.open(next((art / label).glob('*.png'))).convert('RGB')
    picture.thumbnail((512, 512))
    canvas.paste(picture, (x+4, y+32))
canvas.save(out / 'image-comparison.png')
candidate_pixels = np.asarray(Image.open(next((art / 'qkv-final-fullwarm-r2-candidate-b1').glob('*.png'))).convert('RGB'))
Image.fromarray(np.abs(reference.astype(np.int16)-candidate_pixels.astype(np.int16)).astype(np.uint8)).save(out / 'lossless-absolute-difference.png')

import torch
environment = dict(torch=torch.__version__, torch_git=torch.version.git_version, cuda=torch.version.cuda,
    versions={k:importlib.metadata.version(k) for k in ['triton', 'transformers', 'diffusers', 'sglang-kernel']},
    gpu_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv,noheader,nounits'], text=True),
    used_gpu_indices=[0,1], tp=1, sp=1, cfg_parallel_size=2, quality='lossless', dtype='bfloat16', mode='eager',
    residency='DiT, text encoder, VAE resident; manual performance mode; no torch.compile',
    request=dict(width=1024, height=1024, num_frames=1, steps=40, cfg=4, seed=42, prompt='Make the cat wear a red hat',
        input_image_sha256=hashlib.sha256((root / 'artifacts/input-media/longlive2-cat.png').read_bytes()).hexdigest()),
    headline_warmup='same-shape native request warmup, warmup_steps=40 in both arms')
(art / 'environment.json').write_text(json.dumps(environment, indent=2))
print(json.dumps(dict(qualified=report['qualified'], groups={v:{r:g['means'] for r,g in gs.items()} for v,gs in comparisons.items()}, outputs=len(outputs), high=high, profiles=profiles), indent=2), flush=True)
