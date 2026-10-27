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


def run(source, label, qualities):
    arm = 'omni' if source.startswith('omni') else 'sgl'
    assert idle(), 'GPU ownership preflight failed'
    name = f'joy-{label}-{source}'
    folder = ROOT / 'block-runs' / name
    folder.mkdir(exist_ok=False)
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    script = {'omni': 'launch_omni_joy_qualification.sh', 'omni_compile': 'launch_omni_joy_compile.sh', 'omni_compile_casts': 'launch_omni_joy_compile_casts.sh'}[source]
    repo = '/task/candidates/joy-image-gelu' if source == 'pr' else '/task/repos/' + ('sglang' if arm == 'sgl' else 'vllm-omni')
    pid_path = f'/task/block-runs/{name}/server.pid'
    launch = ['docker', 'exec', '-e', f'SGL_REPO={repo}', container, 'sh', '-c', f'echo $$ > {pid_path}; exec sh /task/{script}']
    sha = command(['docker', 'exec', '-w', repo, container, 'git', 'rev-parse', 'HEAD'])
    diff = subprocess.check_output(['docker', 'exec', '-w', repo, container, 'git', 'diff', 'HEAD', '--binary', '--full-index'])
    if source != 'pr':
        assert not diff, 'Baseline tracked source modified'
    else:
        assert diff, 'Candidate patch absent'
    (folder/'source.patch').write_bytes(diff)
    metadata = dict(arm=arm, source=source, repo=repo, label=label, benchmark=False, source_sha=sha, launch_command=launch,
                    diff_sha256=hashlib.sha256(diff).hexdigest(),
                    contract_sha256=hashlib.sha256((ROOT/'inputs/joy-workloads.json').read_bytes()).hexdigest(),
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
        for quality in qualities:
            request = ['docker', 'exec', container, 'python', '/task/submit_joy_candidate_qualification.py', arm,
                       '--label', f'{label}-{source}-{quality}', '--quality', quality]
            metadata['qualification_commands'].append(request)
            (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
            with (folder/f'qualification-{quality}.log').open('wb') as stream:
                subprocess.run(request, stdout=stream, stderr=subprocess.STDOUT, check=True)
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
    p.add_argument('--arms', nargs='+', choices=['omni', 'omni_compile', 'omni_compile_casts'], default=['omni_compile', 'omni_compile_casts'])
    p.add_argument('--wait-download', action='store_true')
    p.add_argument('--qualities', nargs='+', choices=['high', 'lossless', 'exact'], default=['high'])
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
        run(arm, a.label, a.qualities)
