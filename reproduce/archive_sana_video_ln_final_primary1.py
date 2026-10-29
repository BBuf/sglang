"""Preserve every completed final-head measurement, even a failed performance gate."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
run_id = 'sana-video-ln-final-primary-cleanup1'
status = json.loads((root/(run_id+'-status.json')).read_text())
assert status['status'] == 'raw_complete_audit_pending' and len(status['completed']) == 12
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
admission = root/'evidence/preflight'/f'{run_id}-admission.json'
assert admission.exists(), 'Archive successful or failed gates without choosing samples'
paths = {admission}
for area in ['outputs/benchmarks', 'block-runs']:
    for folder in (root/area).glob(run_id+'-*'):
        paths.update(path for path in folder.rglob('*') if path.is_file())
for pattern in [run_id+'*.json', run_id+'*.log', '*sana_video*1.py', '*sana_video*1.sh']:
    paths.update(path for path in root.glob(pattern) if path.is_file())
paths.add(root/'inputs/sana-video-480p-t2v-workloads.json')
paths = sorted(paths)
assert sum(path.name == 'output.mp4' for path in paths) == 96
assert all('cache' not in path.relative_to(root).parts and not path.is_symlink() for path in paths)
name = 'sana-video-ln-final-primary1-archive'
archive = root/(name+'.tar.gz')
assert not archive.exists()
members = [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
with tarfile.open(archive, 'w:gz', compresslevel=3) as stream:
    for path in paths:
        stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
result = dict(archive=archive.name, bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=members)
(root/(name+'-manifest.json')).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key != 'members'}), len(members), flush=True)
