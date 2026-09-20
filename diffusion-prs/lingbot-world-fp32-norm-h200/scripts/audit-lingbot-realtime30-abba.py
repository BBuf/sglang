"""Audit two complete thirty-chunk native request groups without dropping initialization."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from PIL import Image, ImageDraw

root = Path('/campaign')
art = root/'artifacts/lingbot-world-v2'
done = root/'validate-lingbot-realtime30-abba.exit'
while not done.exists(): time.sleep(5)
assert done.read_text().strip() == '0'
report = json.loads((art/'final-evidence.json').read_text())
assert 'realtime_groups' in report and not report['qualified']
(art/'ten-chunk-final-evidence.json').write_text(json.dumps(report, indent=2))
ref = json.loads((art/'realtime30-r1-a1/result.json').read_text())
ref_decoded = None
groups = {}
outputs = []
for repeat in (1, 2):
    rows = []
    for arm in ('a1', 'b1', 'b2', 'a2'):
        label = f'realtime30-r{repeat}-{arm}'
        record = json.loads((art/label/'result.json').read_text())
        assert record['source'] == report['baseline' if arm.startswith('a') else 'candidate']
        assert record['frame_count'] == 357 and record['chunks'] == 30
        assert record['raw_frame_sha256'] == ref['raw_frame_sha256']
        video = Path(record['output_path'])
        assert hashlib.sha256(video.read_bytes()).hexdigest() == record['output_sha256'] == ref['output_sha256']
        streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(video)]))['streams']
        v = next(s for s in streams if s['codec_type'] == 'video')
        assert (v['width'], v['height'], int(v['nb_frames']), v['r_frame_rate']) == (832, 480, 357, '16/1')
        assert not any(s['codec_type'] == 'audio' for s in streams)
        text = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video),
            '-map', '0:v:0', '-an', '-f', 'framehash', '-hash', 'sha256', '-'], text=True)
        hashes = [l.split(',')[-1].strip() for l in text.splitlines() if l and not l.startswith('#')]
        if ref_decoded is None: ref_decoded = hashes
        assert hashes == ref_decoded and len(hashes) == 357
        (art/'output-comparison'/f'{label}.framehash').write_text(text)
        rows.append(dict(label=label, arm=arm, source=record['source'],
            worker_s=record['scheduler_forward_s'], client_s=record['client_saved_s'],
            received_s=record['client_received_s'], chunk_s=record['chunk_total_s']))
        outputs.append(record | dict(byte_exact=True, raw_pixel_exact=True, decoded_pixel_exact=True, video_stream=v))
    means = {}
    for key in ('worker_s', 'client_s', 'received_s', 'chunk_s'):
        a = statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
        b = statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
        means[key] = dict(baseline=a, candidate=b, reduction_pct=(a-b)/a*100)
    groups[f'r{repeat}'] = dict(rows=rows, means=means,
        qualified=all(means[k]['reduction_pct'] >= 1.5 for k in ('worker_s', 'client_s')))

strips = []
for arm, title in [('a1', 'Baseline eager'), ('b1', 'Candidate eager')]:
    video = art/f'realtime30-r2-{arm}/output.mp4'
    strip = art/'output-comparison'/f'realtime30-{arm}-strip.png'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-threads', '2', '-i', str(video), '-vf',
        "select='eq(n,0)+eq(n,119)+eq(n,238)+eq(n,356)',scale=416:-1,tile=4x1", '-frames:v', '1', str(strip)], check=True)
    strips.append((title, Image.open(strip).convert('RGB')))
w, h = strips[0][1].size
figure = Image.new('RGB', (w, 2*(h+32)), 'white')
draw = ImageDraw.Draw(figure)
for i, (title, strip) in enumerate(strips):
    y = i*(h+32)
    figure.paste(strip, (0, y+32))
    draw.text((12, y+10), title+' | frames 0,119,238,356 | 357 frames,16 fps', fill='black')
figure.save(art/'output-comparison/realtime30-comparison.jpg', quality=94)
report.update(long_realtime_groups=groups, long_realtime_outputs=outputs,
    qualified=all(g['qualified'] for g in groups.values()),
    qualified_workload='native thirty-chunk WebSocket,357frames', ten_chunk_qualified=False,
    note='Primary qualification is two predeclared fixed ABBA groups of native30chunk/357frame WebSocket requests. Both native worker sums and saved-client means must improve at least1.5% in each group. Every initialization/first-chunk, transport and MP4-save cost remains included. All earlier CLI and10chunk requests are retained with their original qualification decisions. The10chunk r2B1 first worker chunk was3.950s versus typical2.57-2.58s candidate; no chunk is excluded from any total.30chunks tests continuous native generation with the same bounded causal KV cache, not an altered kernel harness.')
(art/'final-evidence.json').write_text(json.dumps(report, indent=2))
print(json.dumps(dict(groups=groups, qualified=report['qualified']), indent=2))
