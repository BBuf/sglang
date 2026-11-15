"""Archive completed Nomas follow-up runs only after the serial GPU queue ends."""
import datetime
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

root = Path('/task')
preflight = root / 'evidence/preflight'
assert json.loads((preflight / 'qwen21-repeat34b-controller.json').read_text())['status'] == 'complete'
audio = json.loads((preflight / 'ltx23-audio39-controller.json').read_text())
assert audio['status'] == 'raw_complete_requires_audio_repeat_audit' or audio['status'].startswith('failed')
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert json.loads((preflight / 'nomas-repeat33b-controller.json').read_text())['status'] == 'complete'
out = root / 'evidence/nomas-followup41-raw.tar'
receipt = out.with_suffix('.receipt.json')
assert not out.exists() and not receipt.exists()
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
queue = json.loads((root / 'inputs/nomas-repeat33-queue.json').read_text())
scope = json.loads((root / 'inputs/nomas-long-scope29-queue.json').read_text())
visits = queue['visits'] + scope
runs = [v['run'] for v in visits]
paths = set()
excluded_server_copies = []
for name in runs:
    folder = root / 'runs' / name
    state = json.loads((folder / 'status.json').read_text())
    assert state['status'] == 'raw_complete' and state['gpu_cleanup'] == 'idle_verified'
    for path in folder.rglob('*'):
        if path.is_file():
            assert not path.is_symlink(), path
            if 'server-storage' in path.parts:
                excluded_server_copies.append(str(path.relative_to(root)))
            else:
                paths.add(path)
paths.update(root / v['plan'] for v in visits)
paths.update(root / 'inputs' / n for n in ('nomas-repeat33-queue.json', 'nomas-long-scope29-queue.json', 'nomas-sp2-deploy18.yaml'))
for folder in {Path(v['script']).parent for v in visits}:
    paths.update((root / folder).glob('*.py'))
for name in ('nomas-sp2-observer18', 'nomas-sp2-candidate-observer18'):
    paths.update((root / 'scripts' / name).glob('*.py'))
for name in ('run_nomas_long_scope29.py', 'run_nomas_repeat33.py', 'run_nomas_repeat33b.py', 'analyze_protocol_rounds35.py', 'archive_nomas_followup41.py'):
    paths.add(root / 'scripts' / name)
for pattern in ('nomas*29*.json', 'nomas*33*.json', 'nomas-three-rounds35-analysis.log'):
    paths.update(preflight.glob(pattern))
paths.update([root / 'evidence/quality/nomas-long-scope29-summary36.json', root / 'evidence/benchmarks/nomas-three-rounds35.json'])
assert all(p.is_file() for p in paths)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
files = []
with tarfile.open(out, 'x') as tar:
    for path in sorted(paths):
        rel = str(path.relative_to(root))
        files.append(dict(path=rel, bytes=path.stat().st_size, sha256=sha(path)))
        tar.add(path, arcname=rel, recursive=False)
data = dict(started_at=started, completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            archive=str(out), bytes=out.stat().st_size, sha256=sha(out), files=files,
            runs=runs, excluded_redundant_server_storage=excluded_server_copies,
            note='Downloaded outputs and native observations are included. Server-owned duplicate staging paths remain on the task machine. No weights or shared files included.')
receipt.write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps({k: data[k] for k in ('archive', 'bytes', 'sha256', 'completed_at')}) + '\nfiles=' + str(len(files)))
