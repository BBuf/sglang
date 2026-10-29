"""Serial native baseline qualification and separate SGL profiles; no speedup claim."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
RUN = 'sana-video-depthwise-v1-qualification1'
STATUS = ROOT / (RUN + '-status.json')
completed = []


def command(args):
    return subprocess.check_output(args, text=True).strip()


def save(status, **fields):
    STATUS.write_text(json.dumps(dict(status=status, completed=completed, benchmark=False, **fields), indent=2) + '\n')


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def run(arm):
    assert idle(), 'Unowned GPU work present; no server will be started'
    preflight = command(['docker', 'exec', '-e', 'PYTHONPATH=/task/repos/sglang/python', 'bbuf-diffusion-rtx6000-20261026-sgl', 'python3', '/task/preflight_ming_environment.py'])
    print(preflight, flush=True)
    name = RUN + '-' + arm
    folder = ROOT / 'block-runs' / name
    folder.mkdir(exist_ok=False)
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    assert arm == 'sgl'
    repo = '/task/candidates/sana-video-depthwise-glu-v1'
    expected = 'af21e2b989e482cfdce39689126562f2209d53ac' if arm == 'sgl' else '817d4c17feb702289a2d2b2cbacfd594c262b7a7'
    assert command(['docker', 'exec', container, 'git', '-C', repo, 'rev-parse', 'HEAD']) == expected
    patch = subprocess.check_output(['docker', 'exec', container, 'git', '-C', repo, 'diff', 'HEAD', '--binary', '--full-index'])
    assert hashlib.sha256(patch).hexdigest() == '6a073c8df5e246ea84485ac999a3eefc0b2c8949f67b682a85c579d46b345291'
    script = 'launch_sgl_sana_video_depthwise_v1.sh'
    launch = ['docker', 'exec', '-e', f'SGLANG_DIFFUSION_TORCH_PROFILER_DIR=/task/outputs/profiles/{name}',
              container, 'sh', '-c', f'echo $$ > /task/block-runs/{name}/server.pid; exec sh /task/{script}']
    metadata = dict(arm=arm, source_sha=expected, source_diff_sha256=hashlib.sha256(patch).hexdigest(),
                    benchmark=False, launch_command=launch, requests=[], physical_gpu_pool=[0,1], physical_gpus_used=[0],
                    note='Independent frozen-main candidate plus pinned depthwise3x3+biasGLU patch, without LN PR changes. Same native contract; qualification/profile only, no performance claim.',
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder / 'launch.sh').write_bytes((ROOT / script).read_bytes())
    (folder / 'source.patch').write_bytes(patch)
    log = folder / 'server.log'
    save('starting', active=name)
    start = time.monotonic()
    with log.open('xb') as stream:
        server = subprocess.Popen(launch, stdout=stream, stderr=subprocess.STDOUT)
    try:
        port = 31050 if arm == 'sgl' else 31051
        endpoint = '/liveness' if arm == 'sgl' else '/health'
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}: {log}')
            if time.monotonic() - start > 3600:
                raise TimeoutError('Initial native readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}{endpoint}', timeout=2) as response:
                    ready = response.status == 200
            except Exception:
                ready = False
            if arm == 'sgl':
                ready = ready and b'Warmup requests: 100%' in log.read_bytes()
            if ready:
                break
            time.sleep(2)
        if arm == 'sgl':
            assert not any(marker in log.read_text() for marker in ['Falling back to diffusers backend', 'Using diffusers backend', 'Loaded diffusers pipeline']), 'Native backend gate failed'
        metadata['launch_to_ready_seconds'] = time.monotonic() - start
        metadata['server_container_pid'] = int((folder / 'server.pid').read_text())
        metadata['gpu_snapshot_before_requests'] = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'])
        for task in ['t2v']:
            contract = ROOT / 'inputs' / 'sana-video-480p-t2v-workloads.json'
            for profiled in ([False, True] if arm == 'sgl' else [False]):
                label = 'depthwise-v1-native1' + ('-profile' if profiled else '')
                cmd = ['docker', 'exec', container, 'python3', '/task/submit_sana_video_candidate_qualification.py', arm,
                       '--repo', repo, '--expected-diff-sha256', metadata['source_diff_sha256'], '--label', label, '--contract', '/task/inputs/' + contract.name]
                if profiled:
                    cmd.append('--profile')
                metadata['requests'].append(dict(command=cmd, contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(), diagnostic_profile=profiled))
                (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
                save('qualifying' if not profiled else 'profiling', active=name, request_label=label)
                with (folder / (label + '.log')).open('xb') as stream:
                    subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
        if arm == 'sgl':
            assert not any(marker in log.read_text() for marker in ['Falling back to diffusers backend', 'Using diffusers backend', 'Loaded diffusers pipeline']), 'Native backend gate failed'
            for perf in (ROOT / 'outputs/qualification/sgl-sana-video-depthwise-v1-native1').glob('*/perf.json'):
                assert len(json.loads(perf.read_text())['denoise_steps_ms']) == 50, str(perf)
        metadata['status'] = 'generated_raw_quality_and_diagnostics'
    except Exception as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    finally:
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        pid_file = folder / 'server.pid'
        if server.poll() is None and pid_file.exists():
            subprocess.run(['docker', 'exec', container, 'kill', '-TERM', pid_file.read_text().strip()], check=True)
        server.wait(timeout=60)
        for _ in range(30):
            if idle():
                break
            time.sleep(1)
        assert idle(), 'Remaining GPU processes prohibit another arm'
    completed.append(name)


if __name__ == '__main__':
    assert not STATUS.exists(), 'No automatic reruns or output overwrite'
    try:
        plan = json.loads((ROOT / 'sana-video-depthwise-v1-qualification1-plan.json').read_text())
        for path, expected_hash in plan['sha256'].items():
            actual = ROOT / (Path(path).name if path.startswith('scripts/') else path)
            assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected_hash, path
        assert json.loads((ROOT / 'evidence/preflight/sana-video-noise-schedule1.json').read_text())['all_exact']
        manifest = json.loads((ROOT / 'evidence/downloads/sana-video-480p-modelscope-download-manifest.json').read_text())
        assert manifest['complete'] and manifest['hf_revision'] == 'db5f398b13ca086d09a50ce156c20527773841b1'
        assert len(manifest['files']) == 18
        audit = json.loads((ROOT/'evidence/preflight/sana-video-runtime-input-audit1.json').read_text())
        assert audit['observed_outputs_byte_exact'] and audit['noise_and_schedule_exact'] and audit['token_ids_masks_exact']
        for arm in ['sgl']:
            run(arm)
        comparisons=[]
        original=ROOT/'outputs/qualification/sgl-sana-video-t2v-native1'
        for label in ['depthwise-v1-native1','depthwise-v1-native1-profile']:
            for path in sorted((ROOT/'outputs/qualification'/('sgl-sana-video-'+label)).glob('*/output.mp4')):
                exact=path.read_bytes()==(original/path.parent.name/'output.mp4').read_bytes()
                comparisons.append(dict(file=str(path),MP4_exact_main=exact))
        assert len(comparisons)==5 and all(row['MP4_exact_main'] for row in comparisons),comparisons
        save('raw_complete_MP4_exact_main_profile_audit_pending', active=None, comparisons=comparisons)
    except Exception as exc:
        save('failed_preserve_all_outputs_no_retry', error=repr(exc))
        raise
