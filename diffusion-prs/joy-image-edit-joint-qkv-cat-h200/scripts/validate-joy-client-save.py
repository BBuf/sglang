"""Predeclared two fixed ABBA groups, each process one saved warmup + five requests."""
import json
from pathlib import Path
import re
import subprocess
import time

root = Path('/campaign')
art = root / 'artifacts/joy-image-edit'
done = root / 'analyze-joy-request-tail-retry1.exit'
while not done.exists(): time.sleep(5)
assert done.read_text().strip() == '0'
diag = json.loads((art / 'request-tail-diagnostic-final.json').read_text())
assert any(x['seconds'] > .5 and x['name'] == 'torch.cuda.empty_cache' and x['rank'] == '0' for arm in diag.values() for row in arm['rows'].values() for x in row['tail'])
protocol = dict(order=['baseline', 'candidate', 'candidate', 'baseline'], rounds=2,
    saved_warmups_per_process=1, measured_requests_per_process=5, native_warmup_steps=40,
    acceptance='Each round arithmetic mean reduction >=1.5% in worker, native saved-client, and outer saved-client wall. No measured requests discarded.',
    return_file_paths_only=False,
    motivation='Default worker-save oneshot and persistent results are retained and fail saved-client qualification. Separate matched native diagnostics attribute a >0.5 s output-rank tail to empty_cache, including a 1.76 s baseline case. Both unchanged measured revisions now use the existing native API client-save path, which retains output pixel data through transport and saves the PNG within the native client timer. No empty_cache/GC function is patched or disabled, and generation, transport, saving and reporting remain timed. This is an explicitly different supported output configuration, not a claimed default-file-path client improvement.',
    baseline='80da4432d085ed4d6166ef643d9fd2b829dbb0c5', candidate='4811f4d52aa2586412f699b3bb84ed184d76250a')
(art / 'client-save-protocol.json').write_text(json.dumps(protocol, indent=2))
all_rows = []
for round_index in [1, 2]:
    for arm, suffix in [('baseline', 'a1'), ('candidate', 'b1'), ('candidate', 'b2'), ('baseline', 'a2')]:
        label = f'client-save-r{round_index}-{arm}-{suffix}'
        repo = root / ('baseline-super-current' if arm == 'baseline' else 'candidate-joy-outplace-v1')
        log_path = root / f'joy-image-edit-{label}.log'
        with log_path.open('x') as log:
            code = subprocess.run(['python', '-u', str(root / 'run-joy-client-save.py'), '--repo', str(repo), '--label', label], stdout=log, stderr=subprocess.STDOUT).returncode
        (root / f'joy-image-edit-{label}.exit').write_text(str(code))
        assert code == 0, label
        rows = json.loads((art / 'client-save' / label / 'records.json').read_text())
        text = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
        clients = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds', text)
        assert len(rows) == len(clients) == 6, (label, len(rows), len(clients))
        for row, client in zip(rows, clients):
            row.update(label=label, round=round_index, arm=arm, client_saved_s=float(client))
        all_rows.extend(rows)
        (art / 'qkv-abba-client-save-v1.json').write_text(json.dumps(all_rows, indent=2))
        print(json.dumps(dict(label=label, rows=rows)), flush=True)
print('Completed all 48 saved requests; 8 explicit warmups and 40 measured requests retained', flush=True)
