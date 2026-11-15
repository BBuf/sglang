"""Archive the completed Qwen repeats after the queued LTX unit checks."""
import datetime
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def archive():
    root = Path('/task')
    assert json.loads((root / 'evidence/preflight/qwen21-repeat34b-controller.json').read_text())['status'] == 'complete'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    queue = json.loads((root / 'inputs/qwen21-repeat34-queue.json').read_text())
    report = json.loads((root / 'evidence/benchmarks/qwen21-three-rounds35.json').read_text())
    assert report['all_original_round1_samples_retained']
    out = root / 'evidence/qwen21-followup47-raw.tar'
    receipt = out.with_suffix('.receipt.json')
    assert not out.exists() and not receipt.exists()
    paths = set()
    excluded = []
    for visit in queue['visits']:
        folder = root / 'runs' / visit['run']
        state = json.loads((folder / 'status.json').read_text())
        assert state['status'] == 'raw_complete' and state['gpu_cleanup'] == 'idle_verified'
        rows = json.loads((folder / 'outputs/results.json').read_text())
        assert len(rows) == 9 and all(row['matches_reference'] for row in rows)
        paths.add(root / visit['plan'])
        paths.update((root / Path(visit['script']).parent).glob('*.py'))
        for p in folder.rglob('*'):
            if p.is_file():
                if 'server-storage' in p.parts:
                    excluded.append(str(p.relative_to(root)))
                else:
                    assert not p.is_symlink(), p
                    paths.add(p)
    for name in ('run_qwen21_repeat34.py', 'run_qwen21_repeat34b.py', 'analyze_protocol_rounds35.py', 'archive_qwen_followup47.py'):
        paths.add(root / 'scripts' / name)
    for pattern in ('qwen21*34*.json', 'qwen21-three-rounds42-analysis.log'):
        paths.update((root / 'evidence/preflight').glob(pattern))
    paths.update(root / name for name in ('inputs/qwen21-repeat34-queue.json', 'evidence/benchmarks/qwen21-three-rounds35.json'))
    assert all(p.is_file() for p in paths)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    files = []
    with tarfile.open(out, 'x') as tar:
        for p in sorted(paths):
            rel = str(p.relative_to(root))
            files.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha(p)))
            tar.add(p, arcname=rel, recursive=False)
    data = dict(started_at=started, completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                archive=str(out), bytes=out.stat().st_size, sha256=sha(out), files=files,
                runs=[v['run'] for v in queue['visits']], excluded_redundant_server_storage=excluded,
                note='All downloaded outputs and request/response records included. Duplicate server staging remains on the task host. No weights.')
    receipt.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k: data[k] for k in ('archive', 'bytes', 'sha256', 'completed_at')}))

def wait_and_archive():
    root = Path('/data/xyzhang/diffusion-6000-20261113')
    path = root / 'evidence/preflight/qwen21-followup47-controller.json'
    assert not path.exists()
    state = dict(status='waiting_ltx_units45', benchmark=False)
    def save(**values):
        state.update(values, at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        path.write_text(json.dumps(state, indent=2) + '\n')
    try:
        save()
        deadline = time.monotonic() + 5 * 3600
        while True:
            prior = json.loads((root / 'evidence/preflight/ltx23-gelu-units45-controller.json').read_text())
            if prior['status'].startswith(('complete_', 'failed_')):
                break
            assert time.monotonic() < deadline, 'units45 deadline'
            time.sleep(15)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        command = ['docker', 'exec', 'bbuf-diffusion-6000-20261113-sgl', '/opt/sglang/bin/python', '/task/scripts/archive_qwen_followup47.py', 'archive']
        save(status='archiving', command=command)
        with (root / 'evidence/preflight/qwen21-followup47-archive.log').open('x') as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
        save(status='complete_archive_requires_local_verification_and_public_update')
    except BaseException as exc:
        save(status='failed_preserve_no_automatic_retry', error=repr(exc))
        raise

if len(sys.argv) == 2 and sys.argv[1] == 'archive':
    archive()
else:
    assert len(sys.argv) == 1
    wait_and_archive()
