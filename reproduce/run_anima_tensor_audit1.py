"""Serial native baseline qualification and separate SGL profiles; no speedup claim."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
RUN = 'anima-tensor-audit1'
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
    repo = '/task/repos/' + ('sglang' if arm == 'sgl' else 'vllm-omni')
    expected = 'af21e2b989e482cfdce39689126562f2209d53ac' if arm == 'sgl' else '817d4c17feb702289a2d2b2cbacfd594c262b7a7'
    assert command(['docker', 'exec', container, 'git', '-C', repo, 'rev-parse', 'HEAD']) == expected
    patch = subprocess.check_output(['docker', 'exec', container, 'git', '-C', repo, 'diff', 'HEAD', '--binary', '--full-index'])
    assert not patch
    script = 'launch_sgl_anima_audit1.sh' if arm == 'sgl' else 'launch_omni_anima_audit1.sh'
    launch = ['docker', 'exec', '-e', f'SGLANG_DIFFUSION_TORCH_PROFILER_DIR=/task/outputs/profiles/{name}',
              container, 'sh', '-c', f'echo $$ > /task/block-runs/{name}/server.pid; exec sh /task/{script}']
    metadata = dict(arm=arm, source_sha=expected, source_diff_sha256=hashlib.sha256(patch).hexdigest(),
                    benchmark=False, launch_command=launch, requests=[], physical_gpu_pool=[0,1], physical_gpus_used=[0],
                    note='Opt-in import observer outside compiled modules; record tokens, native conditioner input/output, first/last scheduler step and first SGL rope operands. Require byte-exact original outputs, no timing claim.',
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
        for task in ['t2i']:
            contract = ROOT / 'inputs' / 'anima-t2i-workloads.json'
            for profiled in [False]:
                label = task + '-audit1' + ('-profile' if profiled else '')
                cmd = ['docker', 'exec', container, 'python3', '/task/submit_anima_qualification.py', arm,
                       '--label', label, '--contract', '/task/inputs/' + contract.name]
                if profiled:
                    cmd.append('--profile')
                metadata['requests'].append(dict(command=cmd, contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(), diagnostic_profile=profiled))
                (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
                save('qualifying' if not profiled else 'profiling', active=name, request_label=label)
                with (folder / (label + '.log')).open('xb') as stream:
                    subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
        if arm == 'sgl':
            assert not any(marker in log.read_text() for marker in ['Falling back to diffusers backend', 'Using diffusers backend', 'Loaded diffusers pipeline']), 'Native backend gate failed'
            for perf in (ROOT / 'outputs/qualification/sgl-anima-t2i-audit1').glob('*/perf.json'):
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
        for name, key in [('anima-checkpoint-equivalence1.json', 'all_tensors_equal'),
                          ('anima-noise-schedule1.json', 'all_exact')]:
            assert json.loads((ROOT / 'evidence/preflight' / name).read_text())[key]
        for arm in ['sgl', 'omni']:
            run(arm)
        comparisons = []
        for arm in ['sgl', 'omni']:
            observed = ROOT / 'outputs/qualification' / f'{arm}-anima-t2i-audit1'
            original = ROOT / 'outputs/qualification' / f'{arm}-anima-t2i-native1'
            for p in sorted(observed.glob('*/output.png')):
                q = original / p.relative_to(observed)
                comparisons.append(dict(observed=str(p), reference=str(q),
                    observed_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    reference_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),
                    byte_exact=p.read_bytes()==q.read_bytes()))
        assert len(comparisons) == 8
        save('raw_complete' if all(r['byte_exact'] for r in comparisons) else 'observer_changed_outputs_investigate',
             active=None, comparisons=comparisons, input_audit_pending=True)
    except Exception as exc:
        save('failed_preserve_all_outputs_no_retry', error=repr(exc))
        raise
