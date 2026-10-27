"""Archive only completed quality stages, including every actual output and gate."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('sources', nargs='+', choices=['main', 'pr', 'omni'])
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
remote_root = '/data/xyzhang/diffusion-rtx6000-20261026'
remote = 'SGL-RTX6000-01:' + remote_root + '/'
status_bytes = subprocess.check_output(['ssh', 'SGL-RTX6000-01', 'cat',
                                      remote_root + '/joy-gelu-v4-qualification1-status.json'])
status = json.loads(status_bytes)
for source in a.sources:
    assert source in status['completed'], 'Never archive an incomplete stage as completed'
    stage = next(s for s in status['plan']['order'] if s['source'] == source)
    arm = 'omni' if source == 'omni' else 'sgl'
    names = [f'{arm}-joy-gelu-v4-{shape}-{source}-{quality}'
             for shape in stage['shapes'] for quality in stage['qualities']]
    gates = [c['candidate']+'-quality.json' for c in status['checks'] if c['candidate'] in names]
    for paths, destination in [(['outputs/qualification/'+n for n in names+gates], 'qualification'),
                               ([f'block-runs/joy-gelu-v4-{source}'], 'block-runs')]:
        subprocess.run(['rsync', '-a', '-e', 'ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3',
                        *[remote+x for x in paths], str(root/'evidence'/destination)+'/'], check=True)
    for name in names:
        folder = root/'evidence/qualification'/name
        rows = json.loads((folder/'results.json').read_text())
        assert len(rows) == 4
        for row in rows:
            path = folder/Path(row['request']).name/'output.png'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    print(json.dumps(dict(source=source, archived_outputs=4*len(names), sha256_verified=True)), flush=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
(root/f'evidence/runner-logs/joy-gelu-v4-qualification1-status-{stamp}.json').write_bytes(status_bytes)
