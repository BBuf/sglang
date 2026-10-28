"""Serial native baseline qualification and separate SGL profiles; no speedup claim."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
RUN = 'qwen21-numerics1'
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
    name = RUN + '-' + arm
    folder = ROOT / 'block-runs' / name
    folder.mkdir(exist_ok=False)
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    repo = '/task/diagnostics/qwen21-numerics-' + arm
    expected = 'af21e2b989e482cfdce39689126562f2209d53ac' if arm == 'sgl' else '817d4c17feb702289a2d2b2cbacfd594c262b7a7'
    assert command(['docker', 'exec', container, 'git', '-C', repo, 'rev-parse', 'HEAD']) == expected
    patch = subprocess.check_output(['docker', 'exec', container, 'git', '-C', repo, 'diff', 'HEAD', '--binary', '--full-index'])
    assert hashlib.sha256(patch).hexdigest() == json.loads((ROOT/"qwen21-numerics-source-patches.json").read_text())[arm]
    script = 'launch_sgl_qwen21_numerics.sh' if arm == 'sgl' else 'launch_omni_qwen21_numerics.sh'
    launch = ['docker', 'exec', '-e', f'SGLANG_DIFFUSION_TORCH_PROFILER_DIR=/task/outputs/profiles/{name}',
              container, 'sh', '-c', f'echo $$ > /task/block-runs/{name}/server.pid; exec sh /task/{script}']
    metadata = dict(arm=arm, source_sha=expected, source_diff_sha256=hashlib.sha256(patch).hexdigest(),
                    benchmark=False, launch_command=launch, requests=[],
                    note='Observation-only instrumentation with native arithmetic and fast paths unchanged; PNG compression level 0 on both arms; no timing admission.',
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
            if time.monotonic() - start > 1800:
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
        metadata['launch_to_ready_seconds'] = time.monotonic() - start
        metadata['server_container_pid'] = int((folder / 'server.pid').read_text())
        metadata['gpu_snapshot_before_requests'] = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'])
        for task in ['t2i', 'edit']:
            contract = ROOT / 'inputs' / f'qwen21-base-{task}-workloads.json'
            for profiled in [False]:
                label = 'base-' + task + '-numerics1' + ('-profile' if profiled else '')
                cmd = ['docker', 'exec', container, 'python3', '/task/submit_qwen21_numerics.py', arm,
                       '--label', label, '--quality', 'lossless', '--contract', '/task/inputs/' + contract.name]
                if profiled:
                    cmd.append('--profile')
                metadata['requests'].append(dict(command=cmd, contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(), diagnostic_profile=profiled))
                (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
                save('qualifying' if not profiled else 'profiling', active=name, request_label=label)
                with (folder / (label + '.log')).open('xb') as stream:
                    subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
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
        manifest = json.loads((ROOT / 'evidence/downloads/qwen21-modelscope-download-manifest.json').read_text())
        assert manifest['complete'] and len(manifest['files']) == 24
        assert manifest['hf_revision'] == 'd26bb61231c349cf6b7896fa83353113880e1ba3'
        assert all(row['verified_against_hf'] for row in manifest['files'])
        attribution = json.loads((ROOT / 'evidence/preflight/qwen21-scheduler-rounding-attribution-v2.json').read_text())
        assert all(attribution['controlled_variants_reproduce_native_omni_hashes']['diagnostic_math_exp_only'].values())
        for arm in ['sgl', 'omni']:
            run(arm)
        gates = []
        for task in ['t2i', 'edit']:
            cmd = ['docker', 'exec', 'bbuf-diffusion-rtx6000-20261026-sgl', 'python3', '/task/compare_qwen21_qualification_v2.py',
                   f'omni-qwen21-base-{task}-numerics1', '--reference', f'sgl-qwen21-base-{task}-numerics1', '--output', f'/task/outputs/qualification/qwen21-{task}-numerics1-cross-quality.json']
            with (ROOT / (RUN + '-' + task + '-quality.log')).open('xb') as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            gates.append(dict(task=task, returncode=result.returncode))
        save('raw_complete_numeric_gates_pass_visual_and_profile_audit_pending' if all(x['returncode'] == 0 for x in gates)
             else 'raw_complete_cross_framework_quality_failed_no_performance_admission', active=None, gates=gates)
    except Exception as exc:
        save('failed_preserve_all_outputs_no_retry', error=repr(exc))
        raise
