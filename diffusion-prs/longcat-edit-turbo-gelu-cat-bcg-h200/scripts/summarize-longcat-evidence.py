"""Audit all native saved-image rows, including unsuccessful experiments."""
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

root = Path(__file__).resolve().parent
art = root/'artifacts/longcat-image-edit-turbo'
logs = root/'artifacts/longcat-logs'
groups = {}
for mode in ('eager', 'bcg'):
    for repeat in (1, 2):
        rows = []
        for arm in ('a1', 'b1', 'b2', 'a2'):
            label = f'combined-{mode}-r{repeat}-{arm}'
            path = art/label
            result = json.loads((path/'result.json').read_text())
            assert not result['error'], result
            perf = json.loads((path/f'longcat-image-edit-turbo_{label}.json').read_text())
            log = re.sub(r'\x1b\[[0-9;]*m', '', (logs/f'longcat-{label}.log').read_text())
            clients = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds', log)
            assert len(clients) == 1, label
            stages = {e['name']: e['duration_ms']/1000 for e in perf['steps']}
            rows.append(dict(label=label, arm=arm,
                source=(path/'source.txt').read_text().splitlines()[0],
                engine_e2e_s=result['e2e_latency_s'], client_e2e_s=float(clients[0]),
                denoise_s=result['denoise_latency_s'], stages_s=stages,
                peak_gib=result['peak_memory_gb'],
                graph_capture=result['bcg_capture_detected'],
                output_sha256=result['output_sha256'][0]))
        means = {}
        for key in ('engine_e2e_s', 'client_e2e_s', 'denoise_s'):
            a = statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
            b = statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
            means[key] = dict(baseline=a, candidate=b, reduction_pct=(a-b)/a*100)
        groups[f'{mode}-r{repeat}'] = dict(rows=rows, means=means)

reference_path = next((art/'combined-bcg-r2-a1').glob('*.png'))
reference = np.asarray(Image.open(reference_path).convert('RGB'))
comparison = []
for path in sorted(art.glob('*/result.json')):
    result = json.loads(path.read_text())
    for png in sorted(path.parent.glob('*.png')):
        array = np.asarray(Image.open(png).convert('RGB'))
        exact = array.shape == reference.shape and np.array_equal(array, reference)
        entry = dict(label=path.parent.name, quality=result.get('quality'),
            valid_performance_row=not result.get('error'),
            artifact_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
            image_size=list(Image.open(png).size), pixel_exact=exact)
        if array.shape == reference.shape:
            delta = array.astype(np.float64) - reference.astype(np.float64)
            mse = float(np.mean(delta**2))
            entry.update(max_abs_difference=float(np.max(np.abs(delta))),
                ssim=1.0 if exact else float(structural_similarity(reference, array, channel_axis=2, data_range=255)),
                psnr_db='infinity' if mse == 0 else float(10*np.log10(255**2/mse)))
        if result.get('quality') == 'lossless' and not result.get('error'):
            assert exact, entry
        comparison.append(entry)

assert len({r['artifact_sha256'] for r in comparison
            if r['quality']=='lossless' and r['valid_performance_row']}) == 1

profiles = {}
for arm in ('baseline', 'combined-eager', 'combined-bcg'):
    path = art/f'{arm}-profile/traces'
    evidence = json.loads((path/'denoise-step2-evidence.json').read_text())
    attribution = json.loads((path/'gelu-cat-attribution.json').read_text())
    assert evidence['crossing_kernels'] == 0
    profiles[arm] = dict(evidence=evidence, gelu_cat=attribution)
assert profiles['combined-bcg']['evidence']['graph_launch_api_count'] == 31
graphs = json.loads((art/'combined-bcg-profile/traces/graph-replay-evidence.json').read_text())
assert graphs['graph_launches'] == {'cudaGraphLaunch': 248}

quality = json.loads((art/'quality-matrix.json').read_text())
assert len(quality) == 8
qualified = all(groups[f'bcg-r{repeat}']['means'][key]['reduction_pct'] >= 1.5
                for repeat in (1, 2) for key in ('engine_e2e_s', 'client_e2e_s'))
final = dict(qualified=qualified, benchmark=groups, profile=profiles,
    graph_replay=graphs, output_comparison=comparison, quality_matrix=quality,
    qualification='Repeated original-eager versus combined-BCG saved-request E2E; all rows retained.',
    limitation='Eager r1 includes a 193.7ms text-encoder outlier and a 2.87s saved-client row. Eager r2 has a 5.79s saved-client row despite normal worker timing. Neither group establishes a saved-client eager E2E improvement; all rows are retained.',
    timing_note='Worker perf total_duration_ms precedes output save. Native client log includes saving PNG and rounds to two decimals. Profile requests excluded.')
(art/'final-evidence.json').write_text(json.dumps(final, indent=2))

out = art/'output-comparison'
out.mkdir(exist_ok=True)
(out/'comparison.json').write_text(json.dumps(comparison, indent=2))
panels = [('Input', root/'artifacts/input-media/longcat-edit-input.jpg'),
          ('Baseline eager', reference_path),
          ('Optimized BCG (identical pixels)', next((art/'combined-bcg-r2-b1').glob('*.png')))]
canvas = Image.new('RGB', (3*632, 464), 'white')
draw = ImageDraw.Draw(canvas)
for i, (label, path) in enumerate(panels):
    frame = Image.open(path).convert('RGB')
    frame.thumbnail((632, 424))
    canvas.paste(frame, (i*632+(632-frame.width)//2, 40+(424-frame.height)//2))
    draw.text((i*632+16, 12), label, fill='black')
canvas.save(out/'comparison.jpg', quality=95)
print(json.dumps(dict(qualified=qualified, means={k:v['means'] for k,v in groups.items()},
                      compared_images=len(comparison)), indent=2))
