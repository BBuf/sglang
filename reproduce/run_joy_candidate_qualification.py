"""Serialize native baseline qualification on GPU0; never benchmark these runs."""
import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def run(source, label, qualities, contracts, candidate_sha='af21e2b989e482cfdce39689126562f2209d53ac', candidate_diff_sha256=None, omni_compile=False):
    arm = 'omni' if source == 'omni' else 'sgl'
    assert idle(), 'GPU ownership preflight failed'
    name = f'joy-{label}-{source}'
    folder = ROOT / 'block-runs' / name
    folder.mkdir(exist_ok=False)
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    script = ('launch_omni_joy_compile.sh' if omni_compile else 'launch_omni_joy_qualification.sh') if arm == 'omni' else 'launch_sgl_joy_candidate.sh'
    repo = '/task/candidates/joy-image-gelu' if source == 'pr' else '/task/repos/' + ('sglang' if arm == 'sgl' else 'vllm-omni')
    pid_path = f'/task/block-runs/{name}/server.pid'
    expected_sha = candidate_sha if source == 'pr' else ('817d4c17feb702289a2d2b2cbacfd594c262b7a7' if arm == 'omni' else 'af21e2b989e482cfdce39689126562f2209d53ac')
    launch = ['docker', 'exec', '-e', f'SGL_REPO={repo}', '-e', f'JOY_CANDIDATE_SHA={expected_sha}', container, 'sh', '-c', f'echo $$ > {pid_path}; exec sh /task/{script}']
    sha = command(['docker', 'exec', '-w', repo, container, 'git', 'rev-parse', 'HEAD'])
    diff = subprocess.check_output(['docker', 'exec', '-w', repo, container, 'git', 'diff', 'HEAD', '--binary', '--full-index'])
    assert sha == expected_sha, 'Source head mismatch'
    if source != 'pr':
        assert not diff, 'Baseline tracked source modified'
    else:
        assert hashlib.sha256(diff).hexdigest() == candidate_diff_sha256, 'Candidate patch mismatch'
    (folder/'source.patch').write_bytes(diff)
    metadata = dict(arm=arm, source=source, repo=repo, label=label, benchmark=False, source_sha=sha, launch_command=launch,
                    diff_sha256=hashlib.sha256(diff).hexdigest(),
                    contracts=[dict(path=str(path), sha256=hashlib.sha256((ROOT/path.relative_to('/task')).read_bytes()).hexdigest()) for path in contracts],
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'launch.sh').write_bytes((ROOT/script).read_bytes())
    (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    log = folder / 'server.log'
    start = time.monotonic()
    with log.open('wb') as stream:
        server = subprocess.Popen(launch, stdout=stream, stderr=subprocess.STDOUT)
    try:
        port = 31040 if arm == 'sgl' else 31041
        path = '/liveness' if arm == 'sgl' else '/health'
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}: {log}')
            if time.monotonic()-start > 1200:
                raise TimeoutError('readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=2) as r:
                    ready = r.status == 200
            except Exception:
                ready = False
            if arm == 'sgl':
                ready = ready and b'Warmup requests: 100%' in log.read_bytes()
            if ready:
                break
            time.sleep(2)
        metadata['launch_to_ready_seconds'] = time.monotonic()-start
        metadata['server_container_pid'] = int((folder/'server.pid').read_text())
        metadata['gpu_processes_before_requests'] = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'])
        metadata['qualification_commands'] = []
        for contract in contracts:
            shape_label = contract.stem.removeprefix('joy-workloads').strip('-') or 'square'
            for quality in qualities:
                request = ['docker', 'exec', container, 'python', '/task/submit_joy_candidate_qualification.py', arm,
                           '--label', f'{label}-{shape_label}-{source}-{quality}', '--quality', quality, '--contract', str(contract)]
                metadata['qualification_commands'].append(request)
                (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
                with (folder/f'qualification-{shape_label}-{quality}.log').open('wb') as stream:
                    subprocess.run(request, stdout=stream, stderr=subprocess.STDOUT, check=True)
                output = ROOT/'outputs/qualification'/f'{arm}-joy-{label}-{shape_label}-{source}-{quality}'
                rows = json.loads((output/'results.json').read_text())
                assert len(rows) == 4
                for row in rows:
                    image = ROOT/Path(row['request']).relative_to('/task')/'output.png'
                    assert hashlib.sha256(image.read_bytes()).hexdigest() == row['sha256']
                if arm == 'sgl':
                    assert rows[0]['sha256'] == rows[-1]['sha256'], 'Short-repeat output changed'
        metadata['status'] = 'quality_outputs_generated_not_yet_compared'
        print(json.dumps({'block': name, 'status': metadata['status']}), flush=True)
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = repr(exc)
        raise
    finally:
        (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        pid_file = folder/'server.pid'
        if server.poll() is None and pid_file.exists():
            subprocess.run(['docker', 'exec', container, 'kill', '-TERM', pid_file.read_text().strip()], check=True)
        server.wait(timeout=60)
        for _ in range(30):
            if idle():
                break
            time.sleep(1)
        assert idle(), 'GPU processes remain; next block prohibited'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--arms', nargs='+', choices=['main', 'pr', 'omni'], default=['main', 'pr'])
    p.add_argument('--wait-download', action='store_true')
    p.add_argument('--contracts', nargs='+', type=Path, default=[Path('/task/inputs/joy-workloads.json')])
    p.add_argument('--candidate-sha', default='af21e2b989e482cfdce39689126562f2209d53ac')
    p.add_argument('--candidate-diff-sha256')
    p.add_argument('--omni-compile', action='store_true')
    p.add_argument('--qualities', nargs='+', choices=['high', 'lossless', 'exact'], default=['high', 'lossless'])
    a = p.parse_args()
    manifest_path = ROOT/'evidence/downloads/joy-modelscope-download-manifest.json'
    start = time.monotonic()
    while not manifest_path.exists():
        assert a.wait_download and time.monotonic()-start < 3600, 'Verified weights unavailable'
        time.sleep(5)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and len(manifest['files']) == 36
    assert manifest['hf_revision'] == '4b41fb25d961f37668750178ccbb380da326201c'
    assert all(f['verified_against_hf'] for f in manifest['files'])
    for arm in a.arms:
        run(arm, a.label, a.qualities, a.contracts, a.candidate_sha, a.candidate_diff_sha256, a.omni_compile)
