"""Predeclared two fixed ABBA groups, each process one saved warmup + five requests."""
import json
from pathlib import Path
import re
import subprocess

root = Path('/campaign')
art = root / 'artifacts/joy-image-edit'
assert (root / 'validate-joy-outplace-v1.exit').read_text().strip() == '0'
protocol = dict(order=['baseline', 'candidate', 'candidate', 'baseline'], rounds=2,
    saved_warmups_per_process=1, measured_requests_per_process=5, native_warmup_steps=40,
    acceptance='Each round arithmetic mean reduction >=1.5% in worker, native saved-client, and outer saved-client wall. No measured requests discarded.',
    motivation='Oneshot combined round 2 retained a 16.08 s saved-client outlier (13.824 s worker). Builtin request warmup skips non-warmup replica peak-memory collective/reporting. Persistent full saved-request warmup tests steady-state behavior; cause of the outlier is not proven.',
    baseline='80da4432d085ed4d6166ef643d9fd2b829dbb0c5', candidate='4811f4d52aa2586412f699b3bb84ed184d76250a')
(art / 'persistent-protocol.json').write_text(json.dumps(protocol, indent=2))
all_rows = []
for round_index in [1, 2]:
    for arm, suffix in [('baseline', 'a1'), ('candidate', 'b1'), ('candidate', 'b2'), ('baseline', 'a2')]:
        label = f'persistent-r{round_index}-{arm}-{suffix}'
        repo = root / ('baseline-super-current' if arm == 'baseline' else 'candidate-joy-outplace-v1')
        log_path = root / f'joy-image-edit-{label}.log'
        with log_path.open('x') as log:
            code = subprocess.run(['python', '-u', str(root / 'run-joy-persistent.py'), '--repo', str(repo), '--label', label], stdout=log, stderr=subprocess.STDOUT).returncode
        (root / f'joy-image-edit-{label}.exit').write_text(str(code))
        assert code == 0, label
        rows = json.loads((art / 'persistent' / label / 'records.json').read_text())
        text = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
        clients = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds', text)
        assert len(rows) == len(clients) == 6, (label, len(rows), len(clients))
        for row, client in zip(rows, clients):
            row.update(label=label, round=round_index, arm=arm, client_saved_s=float(client))
        all_rows.extend(rows)
        (art / 'qkv-abba-persistent-v1.json').write_text(json.dumps(all_rows, indent=2))
        print(json.dumps(dict(label=label, rows=rows)), flush=True)
print('Completed all 48 saved requests; 8 explicit warmups and 40 measured requests retained', flush=True)
