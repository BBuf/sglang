"""Audit completed Edge requests and media, then remove only its owned weights."""
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
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

root = Path('/campaign')
art = root/'artifacts/cosmos3-edge'
for name in ('validate-cosmos3-combined-retry1', 'cosmos3-edge-unary-compat-tests'):
    done = root/f'{name}.exit'
    while not done.exists():
        time.sleep(5)
    assert done.read_text().strip() == '0', done
head = subprocess.check_output(['git','-C',str(root/'candidate-cosmos3-edge'),'rev-parse','HEAD'], text=True).strip()
assert head == '18f1417a8b189fcb80b38d6ca3b487ecca89f39d'
groups = {}
for mode in ('t2i','t2v'):
    for repeat in (1,2):
        rows = []
        for arm in ('a1','b1','b2','a2'):
            label = f'combined-{mode}-r{repeat}-{arm}'
            result = json.loads((art/label/'result.json').read_text())
            assert not result['error']
            log = re.sub(r'\x1b\[[0-9;]*m', '', (root/f'cosmos3-edge-{label}.log').read_text())
            clients = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds', log)
            assert len(clients) == 1
            source = (art/label/'source.txt').read_text().splitlines()[0]
            assert source == ('993d1fccbaafe3e79d91567d2fc1d665cc94fa50' if arm.startswith('a') else head)
            rows.append(dict(label=label, arm=arm, source=source,
                worker_s=result['e2e_latency_s'], client_s=float(clients[0]),
                denoise_s=result['denoise_latency_s'], peak_gib=result['peak_memory_gb']))
        means = {}
        for key in ('worker_s','client_s','denoise_s'):
            a = statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
            b = statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
            means[key] = dict(baseline=a, candidate=b, reduction_pct=(a-b)/a*100)
        groups[f'{mode}-r{repeat}'] = dict(rows=rows, means=means,
            qualified=all(means[k]['reduction_pct'] >= 1.5 for k in ('worker_s','client_s')))
assert all(groups[f't2i-r{r}']['qualified'] for r in (1,2)), groups

profiles = {}
for label in ('t2i-baseline-profile','t2v-baseline-profile','t1-candidate-profile',
              'combined-t2i-profile','combined-t2v-profile'):
    result = json.loads((art/label/'result.json').read_text())
    assert not result['error'] and result['native_profile_traces']
    source = next((art/label/'traces').glob('*-3_steps-*.trace.json.gz'))
    with (art/label/'step-analysis.json').open('w') as out:
        subprocess.run(['python',str(root/'analyze-cosmos3-trace.py'),str(source)], stdout=out, check=True)
    report = json.loads((source.parent/'denoise-step2-evidence.json').read_text())
    assert report['crossing_kernels'] == 0
    with gzip.open(source.parent/'denoise-step2.trace.json.gz','rt') as stream:
        events = json.load(stream)['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    fused_qk = [e for e in kernels if 'sglang::fused_qknorm_rope_warp' in e['name']]
    fused_relu = [e for e in kernels if 'sglang::act_kernel<' in e['name']]
    relu_ids = {e['args']['External id'] for e in events
        if e.get('name') in ('aten::clamp_min','aten::mul')
        and e.get('args',{}).get('Input Dims',[[]])[0][-1:] == [9216]}
    eager_relu = [e for e in kernels if e.get('args',{}).get('External id') in relu_ids]
    report['activation_attribution'] = dict(
        fused_qk_calls=len(fused_qk), fused_qk_ms=sum(e['dur'] for e in fused_qk)/1000,
        fused_relu_calls=len(fused_relu), fused_relu_ms=sum(e['dur'] for e in fused_relu)/1000,
        eager_relu_calls=len(eager_relu), eager_relu_ms=sum(e['dur'] for e in eager_relu)/1000)
    if label.startswith('combined-'):
        assert len(fused_relu) == 56, (label, len(fused_relu))
    (source.parent/'activation-attribution.json').write_text(json.dumps(report['activation_attribution'], indent=2))
    with (art/label/'triage-step2.txt').open('w') as out:
        subprocess.run(['python',str(root/'llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py'),
            '--framework','sglang','--input',str(source.parent/'denoise-step2.trace.json.gz')], stdout=out, check=True)
    profiles[label] = report

comparison = art/'output-comparison'
comparison.mkdir(exist_ok=True)
ref_image_path = next((art/'combined-t2i-r2-a1').glob('*.png'))
ref_video_path = next((art/'combined-t2v-r2-a1').glob('*.mp4'))
ref_image = np.asarray(Image.open(ref_image_path).convert('RGB'))

def frame_hashes(path):
    text = subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(path),
        '-map','0:v:0','-an','-f','framehash','-hash','sha256','-'], text=True)
    hashes = [line.split(',')[-1].strip() for line in text.splitlines() if line and not line.startswith('#')]
    assert len(hashes) == 81, (path, len(hashes))
    return text, hashes

_, ref_hashes = frame_hashes(ref_video_path)
ref_frames = None
outputs = []
for path in sorted(art.glob('*/result.json')):
    result = json.loads(path.read_text())
    valid = not result['error']
    quality = result['quality']
    if not valid:
        assert result.get('breakable_cuda_graph') and '[diffusion bcg] disabled' in result['bcg_invalid_signals']
    artifacts = result.get('output_artifacts') or [str(p) for p in path.parent.iterdir() if p.suffix in ('.png','.mp4')]
    if not artifacts:
        assert not valid
        outputs.append(dict(label=path.parent.name, quality=quality,
            valid_performance_row=False, artifact_produced=False,
            bcg_invalid_signals=result['bcg_invalid_signals']))
        continue
    assert len(artifacts) == 1
    artifact = Path(artifacts[0])
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if result.get('output_sha256'):
        assert sha == result['output_sha256'][0]
    row = dict(label=path.parent.name, quality=quality, valid_performance_row=valid,
        path=str(artifact), sha256=sha, bcg_capture_detected=result.get('bcg_capture_detected'))
    if artifact.suffix == '.png':
        pixels = np.asarray(Image.open(artifact).convert('RGB'))
        assert pixels.shape == ref_image.shape
        exact = np.array_equal(pixels, ref_image)
        row.update(pixel_exact=exact, width=640, height=640)
        if not exact:
            row['ssim'] = float(structural_similarity(ref_image, pixels, channel_axis=-1, data_range=255))
            row['psnr_db'] = float(peak_signal_noise_ratio(ref_image, pixels, data_range=255))
            assert quality == 'high' and row['ssim'] >= .95 and row['psnr_db'] >= 28, row
    else:
        assert artifact.suffix == '.mp4'
        metadata = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(artifact)]))
        streams = metadata['streams']
        video = next(s for s in streams if s['codec_type'] == 'video')
        assert not any(s['codec_type'] == 'audio' for s in streams)
        assert (video['width'],video['height'],int(video['nb_frames']),video['r_frame_rate']) == (832,480,81,'24/1')
        text, hashes = frame_hashes(artifact)
        (comparison/f'{path.parent.name}.framehash').write_text(text)
        exact = hashes == ref_hashes
        row.update(pixel_exact=exact, frame_count=81, video_stream=video, audio_streams=0)
        if not exact:
            assert quality == 'high', row
            def decode(video_path):
                return np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(video_path),
                    '-map','0:v:0','-an','-f','rawvideo','-pix_fmt','rgb24','-']), dtype=np.uint8).reshape(81,480,832,3)
            if ref_frames is None:
                ref_frames = decode(ref_video_path)
            frames = decode(artifact)
            ssims = [float(structural_similarity(a,b,channel_axis=-1,data_range=255)) for a,b in zip(ref_frames,frames)]
            psnrs = [float(peak_signal_noise_ratio(a,b,data_range=255)) for a,b in zip(ref_frames,frames)]
            row.update(ssim_min=min(ssims), ssim_mean=statistics.mean(ssims), psnr_min_db=min(psnrs), psnr_mean_db=statistics.mean(psnrs))
            assert min(ssims) >= .92 and min(psnrs) >= 24, row
    if quality == 'lossless':
        assert row['pixel_exact'], row
    outputs.append(row)
assert len(outputs) == 39, len(outputs)

figure = Image.new('RGB',(1280,672),'white')
draw = ImageDraw.Draw(figure)
for index, arm in enumerate(('a1','b1')):
    picture = Image.open(next((art/f'combined-t2i-r2-{arm}').glob('*.png'))).convert('RGB')
    figure.paste(picture,(640*index,32))
    draw.text((640*index+12,10), 'Baseline' if arm=='a1' else 'Candidate', fill='black')
figure.save(comparison/'image-comparison.png')
strips = []
for arm in ('a1','b1'):
    video = next((art/f'combined-t2v-r2-{arm}').glob('*.mp4'))
    strip = comparison/f'video-{arm}-strip.png'
    subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-i',str(video),'-vf',
        "select='eq(n,0)+eq(n,26)+eq(n,53)+eq(n,80)',scale=416:-1,tile=4x1",'-frames:v','1',str(strip)],check=True)
    strips.append(Image.open(strip).convert('RGB'))
width,height = strips[0].size
figure = Image.new('RGB',(width,2*(height+32)),'white');draw=ImageDraw.Draw(figure)
for index, strip in enumerate(strips):
    y=index*(height+32)
    figure.paste(strip,(0,y+32));draw.text((12,y+10),('Baseline' if index==0 else 'Candidate')+' | frames 0, 26, 53, 80 | 3.375 s, 24 fps',fill='black')
figure.save(comparison/'video-comparison.jpg',quality=94)
final = dict(qualified=True, candidate=head, groups=groups, profiles=profiles, outputs=outputs,
    note='T2I determines qualification; video gains are reported separately. Every timed row is retained. All BCG requests are explicitly unsupported and excluded. Profile timings are diagnostic only.')
(art/'final-evidence.json').write_text(json.dumps(final,indent=2))
lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
for _ in range(30):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
    if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()):
        break
    time.sleep(1)
else:
    raise RuntimeError('Assigned GPU remains busy; no process was killed')
helper=root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
record=bench._cleanup_model_cache(root/'model-caches',root/'model-caches/cosmos3-edge-cycle',
    root/'artifacts/cache-cleanup.jsonl','cosmos3-edge','cycle',
    'Completed repeated native T2I/T2V ABBA, paired profiles, exact outputs and quality/BCG applicability checks')
assert record['after']==dict(file_count=0,weight_file_count=0,total_bytes=0)
assert not (root/'model-caches/cosmos3-edge-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(groups=groups, output_count=len(outputs), cleanup=record),indent=2))
