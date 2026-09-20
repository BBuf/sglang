"""Validate all predeclared persistent observations; preserve oneshot failures."""
import hashlib
import json
from pathlib import Path
import statistics
import time

root = Path('/campaign')
art = root / 'artifacts/joy-image-edit'
for name in ['audit-joy-outplace-v1-retry1', 'validate-joy-persistent-retry1', 'inspect-joy-copy-qk-chain']:
    marker = root / f'{name}.exit'
    while not marker.exists():
        time.sleep(5)
    assert marker.read_text().strip() == '0', name
e = json.loads((art / 'final-evidence.json').read_text())
assert e['candidate'] == '4811f4d52aa2586412f699b3bb84ed184d76250a'
assert not e['qualified']
(art / 'oneshot-combined-final-evidence.json').write_text(json.dumps(e, indent=2))
rows = json.loads((art / 'qkv-abba-persistent-v1.json').read_text())
assert len(rows) == 48
assert len([r for r in rows if r['role'] == 'measured']) == 40
groups = {}
for round_index in [1, 2]:
    selected = [r for r in rows if r['round'] == round_index]
    assert [r['arm'] for r in selected if r['index'] == 0] == ['baseline', 'candidate', 'candidate', 'baseline']
    for row in selected:
        folder = art / 'persistent' / row['label']
        protocol = json.loads((folder / 'protocol.json').read_text())
        assert protocol['saved_warmups'] == 1 and protocol['measured_requests'] == 5
        assert protocol['native_request_warmup_steps'] == 40
        assert row['source'] == e[row['arm']]
        assert (folder / 'source.txt').read_text().splitlines()[0] == row['source']
        assert row['role'] == ('saved-warmup' if row['index'] == 0 else 'measured')
        assert row['client_wall_s'] >= row['client_saved_s'] - .01
        assert row['worker_s'] > row['denoise_s'] > 0
        image = Path(row['output_file_path'])
        assert image.is_relative_to(folder) and image.is_file()
        assert hashlib.sha256(image.read_bytes()).hexdigest() == row['output_sha256'] == '76cfd40eb83feacb1b0292e171b9beb2a8cd7eb7fadc606cb1632e886d960ce6'
        e['outputs'].append(dict(label=f"{row['label']}/request-{row['index']}", quality='lossless', bcg=False,
            valid_request=True, pixel_exact=True, sha256=row['output_sha256'], ssim=1.0, psnr_db=None,
            peak_memory_gib=row['peak_gib'], source=row['source'], role=row['role']))
    measured = [r for r in selected if r['role'] == 'measured']
    assert len(measured) == 20
    means = {}
    for key in ['worker_s', 'denoise_s', 'client_saved_s', 'client_wall_s']:
        a, b = [statistics.mean(r[key] for r in measured if r['arm'] == arm) for arm in ['baseline', 'candidate']]
        means[key] = dict(baseline=a, candidate=b, reduction_pct=(1-b/a)*100)
    groups[str(round_index)] = dict(rows=measured, saved_warmups=[r for r in selected if r['role'] == 'saved-warmup'], means=means,
        qualifies=all(means[k]['reduction_pct'] >= 1.5 for k in ['worker_s', 'client_saved_s', 'client_wall_s']))
e['comparisons']['persistent-v1'] = groups
e['qualified'] = all(g['qualifies'] for g in groups.values())
e['headline_protocol'] = 'persistent-v1'
e['persistent_protocol'] = json.loads((art / 'persistent-protocol.json').read_text())
e['timing_notes'] += (' Combined oneshot round 2 candidate client 16.08 s outlier is retained and makes that group non-qualifying. '
    'Persistent protocol was declared before running: 2 fixed ABBA groups, each fresh native generator process has native 40-step request warmup, then one complete saved warmup and five measured complete saved requests. '
    'All 40 measured observations and 8 saved warmups retained. Outer wall includes generate and native CLI performance dump; hashing/campaign JSON excluded. No demonstrated root cause for the oneshot outlier.')
e['failed_requests'].append(dict(label='persistent-r1-baseline-a1', reason='Harness called native from_cli_args without replacing sys.argv; model_path absent. Failed before GPU/model initialization; retry1 supplies argv and preserves the failed log.'))
(art / 'final-evidence.json').write_text(json.dumps(e, indent=2))
environment = json.loads((art / 'environment.json').read_text())
environment['headline_warmup'] = '40-step same-shape native warmup + one full saved-output warmup; five measured saved requests per fresh process, two fixed ABBA groups'
(art / 'environment.json').write_text(json.dumps(environment, indent=2))
print(json.dumps(dict(qualified=e['qualified'], means={k:v['means'] for k,v in groups.items()}, outputs=len(e['outputs'])), indent=2), flush=True)
