"""Audit fixed native LingBot comparisons, actual videos, and profile sites."""
import hashlib
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
art = root/'artifacts/lingbot-world-v2'
for name in ('validate-lingbot-current', 'finish-lingbot-current-retry1', 'repeat-lingbot-current-retry1'):
    done = root/f'{name}.exit'
    while not done.exists(): time.sleep(5)
    assert done.read_text().strip() == '0', done
base = 'dd83b54611897a5f80f4df59e69756bd2fb4b8ab'
head = '89b3700dbc9146727cbb368b6859f5a18e79ae66'
groups = {}
for repeat in (1, 2, 3, 4):
    rows = []
    for arm in ('a1', 'b1', 'b2', 'a2'):
        label = f'current-r{repeat}-{arm}'
        record = json.loads((art/label/'result.json').read_text())
        assert not record['error'] and not record['breakable_cuda_graph']
        source = (art/label/'source.txt').read_text().splitlines()[0]
        assert source == (base if arm.startswith('a') else head)
        log = re.sub(r'\x1b\[[0-9;]*m', '', (root/f'lingbot-{label}.log').read_text())
        client = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds', log)
        assert len(client) == 1, label
        rows.append(dict(label=label, arm=arm, source=source,
                         worker_s=record['e2e_latency_s'], client_s=float(client[0]),
                         denoise_s=record['denoise_latency_s'], peak_gib=record['peak_memory_gb']))
    means = {}
    for key in ('worker_s', 'client_s', 'denoise_s'):
        a = statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
        b = statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
        means[key] = dict(baseline=a, candidate=b, reduction_pct=(a-b)/a*100)
    groups[f'r{repeat}'] = dict(rows=rows, means=means,
        qualified=all(means[k]['reduction_pct'] >= 1.5 for k in ('worker_s', 'client_s')))

profiles = {}
for label in ('current-baseline-profile', 'current-profile'):
    trace = next((art/label/'traces').glob('*full_stages*.trace.json.gz'))
    with (trace.parent/'attribution-final.log').open('w') as out:
        subprocess.run(['python', str(root/'analyze-lingbot-trace.py'), str(trace)], stdout=out, check=True)
    sites = json.loads(trace.with_name('lingbot-sites.json').read_text())
    modules = json.loads(trace.with_name('lingbot-attribution.json').read_text())
    assert len(modules['passes']) == 5
    assert sites['sites']['norm1_and_entry_setup']['blocks'] == 200
    assert all(sites['sites'][s]['blocks'] == 199 for s in ('camera', 'residual_before_camera', 'cross_attention_norm'))
    profiles[label] = dict(sites=sites, modules=modules)

comparison = art/'output-comparison'
comparison.mkdir(exist_ok=True)
reference = next((art/'current-r4-a1').glob('*.mp4'))
ref_sha = hashlib.sha256(reference.read_bytes()).hexdigest()

def framehash(path):
    text = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path),
        '-map', '0:v:0', '-an', '-f', 'framehash', '-hash', 'sha256', '-'], text=True)
    return text, [l.split(',')[-1].strip() for l in text.splitlines() if l and not l.startswith('#')]

_, ref_hashes = framehash(reference)
assert len(ref_hashes) == 9
outputs = []
quality_cache = {}
for path in sorted(art.glob('current-*/result.json')):
    if 'realtime' in path.parent.name: continue
    record = json.loads(path.read_text())
    valid = not record['error']
    row = dict(label=path.parent.name, quality=record['quality'], valid_request=valid,
               bcg=record['breakable_cuda_graph'], performance_group=path.parent.name.startswith('current-r'))
    videos = list(path.parent.glob('*.mp4'))
    if not valid:
        assert row['bcg'] and not record['bcg_capture_detected']
        row.update(reason='Native config explicitly disables BCG; any produced eager fallback is invalid BCG evidence',
                   fallback_artifacts=[dict(path=str(v), sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for v in videos])
        outputs.append(row)
        continue
    assert len(videos) == 1
    video = videos[0]
    sha = hashlib.sha256(video.read_bytes()).hexdigest()
    assert record['output_sha256'] == [sha]
    streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(video)]))['streams']
    v = next(s for s in streams if s['codec_type'] == 'video')
    assert not any(s['codec_type'] == 'audio' for s in streams)
    assert (v['width'], v['height'], int(v['nb_frames']), v['r_frame_rate']) == (832, 480, 9, '16/1')
    text, hashes = framehash(video)
    (comparison/f'{path.parent.name}.framehash').write_text(text)
    row.update(path=str(video), sha256=sha, byte_exact=sha == ref_sha,
               pixel_exact=hashes == ref_hashes, frame_count=len(hashes), video_stream=v)
    if record['quality'] == 'lossless':
        assert row['byte_exact'] and row['pixel_exact'], row
    elif sha not in quality_cache:
        def decode(p):
            return np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(p),
                '-map', '0:v:0', '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']), dtype=np.uint8).reshape(9, 480, 832, 3)
        ssims = []
        psnrs = []
        for a, b in zip(decode(reference), decode(video)):
            ssims.append(float(structural_similarity(a, b, channel_axis=-1, data_range=255)))
            psnrs.append(float(peak_signal_noise_ratio(a, b, data_range=255)))
        quality_cache[sha] = dict(ssim_mean=statistics.mean(ssims), ssim_min=min(ssims),
            psnr_mean_db=statistics.mean(psnrs), psnr_min_db=min(psnrs),
            quality_pass=min(ssims) >= .92 and min(psnrs) >= 24)
    if record['quality'] == 'high': row.update(quality_cache[sha])
    outputs.append(row)
high = [r for r in outputs if r['quality'] == 'high' and r['valid_request']]
assert len(high) == 2 and len({r['sha256'] for r in high}) == 1
assert all(r['quality_pass'] for r in high)

realtime = {}
for arm, source in [('baseline', base), ('candidate', head)]:
    record = json.loads((art/('current-realtime-baseline' if arm == 'baseline' else 'current-realtime-candidate-retry1')/'result.json').read_text())
    assert record['source'] == source and record['chunks'] == 10
    video = Path(record['output_path'])
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record['output_sha256']
    text, hashes = framehash(video)
    assert len(hashes) == record['frame_count']
    (comparison/f'realtime-{arm}.framehash').write_text(text)
    record['decoded_frame_sha256'] = hashes
    realtime[arm] = record
assert realtime['baseline']['raw_frame_sha256'] == realtime['candidate']['raw_frame_sha256']
assert realtime['baseline']['decoded_frame_sha256'] == realtime['candidate']['decoded_frame_sha256']
assert realtime['baseline']['output_sha256'] == realtime['candidate']['output_sha256']

for mode in ('cli', 'realtime'):
    n = 9 if mode == 'cli' else realtime['baseline']['frame_count']
    indices = sorted({0, n//3, 2*n//3, n-1})
    strips = []
    for arm, title in [('baseline', 'Baseline eager'), ('candidate', 'Candidate eager')]:
        video = next((art/f'current-r4-{"a1" if arm == "baseline" else "b1"}').glob('*.mp4')) if mode == 'cli' else Path(realtime[arm]['output_path'])
        strip = comparison/f'{mode}-{arm}-strip.png'
        select = '+'.join(f'eq(n,{i})' for i in indices)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-threads', '2', '-i', str(video), '-vf',
            f"select='{select}',scale=416:-1,tile=4x1", '-frames:v', '1', str(strip)], check=True)
        strips.append((title, Image.open(strip).convert('RGB')))
    w, h = strips[0][1].size
    figure = Image.new('RGB', (w, 2*(h+32)), 'white')
    draw = ImageDraw.Draw(figure)
    for i, (title, strip) in enumerate(strips):
        y = i*(h+32)
        figure.paste(strip, (0, y+32))
        draw.text((12, y+10), title+f' | frames {indices} | {n} frames, 16 fps', fill='black')
    figure.save(comparison/f'{mode}-comparison.jpg', quality=94)

history = []
for path in sorted(art.glob('*/result.json')):
    if path.parent.name.startswith('current-'): continue
    record = json.loads(path.read_text())
    history.append(dict(label=path.parent.name, record=record,
        performance_scope='Exploratory or older source; not final-source primary performance evidence',
        requested_117_cli_is_only_9_output_frames=path.parent.name.startswith('full117-')))
report = dict(baseline=base, candidate=head, groups=groups, profiles=profiles, outputs=outputs,
    realtime=realtime, historical_requests=history,
    qualified=all(groups[f'r{r}']['qualified'] for r in (3, 4)),
    note='All four fixed groups retained. R1 contains a 6.06 s candidate client observation after a post-save delay. R3/R4 were predeclared after that observation. Profile/quality/BCG/initial probes are excluded from performance groups. WebSocket is one supplementary native ten-chunk A/B, not a repeated performance claim. CLI --num-frames=117 still emits only 9 frames and is not long-video evidence.')
(art/'final-evidence.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ('profiles', 'outputs', 'historical_requests', 'realtime')}, indent=2))
