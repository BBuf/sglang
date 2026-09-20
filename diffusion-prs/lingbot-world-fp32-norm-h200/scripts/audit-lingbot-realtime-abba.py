"""Qualify repeated native streaming requests while retaining all short-CLI limits."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

root = Path('/campaign')
art = root/'artifacts/lingbot-world-v2'
done = root/'validate-lingbot-realtime-abba.exit'
while not done.exists(): time.sleep(5)
assert done.read_text().strip() == '0'
report = json.loads((art/'final-evidence.json').read_text())
assert not report['qualified'], 'Preserve the earlier CLI qualification decision'
(art/'cli-final-evidence.json').write_text(json.dumps(report, indent=2))
ref = report['realtime']['baseline']
groups = {}
outputs = []
for repeat in (1, 2):
    rows = []
    for arm in ('a1', 'b1', 'b2', 'a2'):
        label = f'realtime-r{repeat}-{arm}'
        record = json.loads((art/label/'result.json').read_text())
        assert record['source'] == report['baseline' if arm.startswith('a') else 'candidate']
        assert record['frame_count'] == 117 and record['chunks'] == 10
        assert record['raw_frame_sha256'] == ref['raw_frame_sha256']
        video = Path(record['output_path'])
        assert hashlib.sha256(video.read_bytes()).hexdigest() == record['output_sha256'] == ref['output_sha256']
        text = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video),
            '-map', '0:v:0', '-an', '-f', 'framehash', '-hash', 'sha256', '-'], text=True)
        hashes = [l.split(',')[-1].strip() for l in text.splitlines() if l and not l.startswith('#')]
        assert hashes == ref['decoded_frame_sha256']
        (art/'output-comparison'/f'{label}.framehash').write_text(text)
        rows.append(dict(label=label, arm=arm, source=record['source'],
            worker_s=record['scheduler_forward_s'], client_s=record['client_saved_s'],
            received_s=record['client_received_s'], chunk_s=record['chunk_total_s']))
        outputs.append(record | dict(byte_exact=True, raw_pixel_exact=True, decoded_pixel_exact=True))
    means = {}
    for key in ('worker_s', 'client_s', 'received_s', 'chunk_s'):
        a = statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
        b = statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
        means[key] = dict(baseline=a, candidate=b, reduction_pct=(a-b)/a*100)
    groups[f'r{repeat}'] = dict(rows=rows, means=means,
        qualified=all(means[k]['reduction_pct'] >= 1.5 for k in ('worker_s', 'client_s')))
report.update(realtime_groups=groups, realtime_outputs=outputs,
    qualified=all(g['qualified'] for g in groups.values()),
    qualified_workload='native ten-chunk WebSocket,117frames',
    short_cli_qualified=False,
    note='Two predeclared fixed ABBA groups of the native ten-chunk WebSocket workload are the qualification basis. Client time includes connection, receiving all raw frames, the native CI MP4 encoding helper, and saving the MP4. Worker time sums the native scheduler_forward_ms chunk statistics. No request timer subtracts transport or saving. All four prior 9-frame CLI groups are retained; their r1 post-save client outlier and r4 sub-threshold client result prevent a stable short-CLI client claim. Initial WebSocket A/B, profiles and quality probes are supplementary only.')
(art/'final-evidence.json').write_text(json.dumps(report, indent=2))
print(json.dumps(dict(groups=groups, qualified=report['qualified']), indent=2))
