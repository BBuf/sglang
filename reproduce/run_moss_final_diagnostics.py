"""Post-pairing mode-switch, final-config traces, and native Omni stage timing.

Run on the remote host only after all 12 predeclared benchmark blocks pass.
These instrumented requests are never part of the primary latency comparison.
"""
import contextlib
import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
HEAD = 'd752066eaa6f5091fa4c8daa2ba1d5d43a3928cd'
BASE = 'af21e2b989e482cfdce39689126562f2209d53ac'
OMNI = '817d4c17feb702289a2d2b2cbacfd594c262b7a7'
RUN = 'moss-qkv-final-diagnostics1'
EMPTY = hashlib.sha256(b'').hexdigest()


def output(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def assert_protocol_complete():
    plan = json.loads((ROOT / 'moss-qkv-final-paired-plan.json').read_text())
    for i, config in enumerate(plan['order']):
        name = f'{plan["id"]}-{i:02d}-{config}'
        folder = ROOT / 'outputs/benchmarks' / name
        summary = json.loads((folder / 'summary.json').read_text())
        metadata = json.loads((folder / 'metadata.json').read_text())
        assert summary['n'] == 4 and summary['all_outputs_byte_identical']
        assert metadata['source_diff_sha256'] == EMPTY
        arm = 'omni' if config == 'omni' else 'sgl'
        quality = json.loads((ROOT / 'outputs/qualification' / f'{arm}-moss-{name}-quality.json').read_text())
        assert quality['all_pass']
    assert idle(), 'Another server or GPU task remains; do not overlap diagnostics.'


@contextlib.contextmanager
def server(config, arm, repo, sha, script, expected_patch_sha256=EMPTY):
    assert idle()
    folder = ROOT / 'block-runs' / f'{RUN}-{config}'
    folder.mkdir(parents=True, exist_ok=False)
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    base = ['docker', 'exec', container]
    def verify_source():
        assert output(base + ['git', '-C', repo, 'rev-parse', 'HEAD']) == sha
        patch = subprocess.check_output(base + ['git', '-C', repo, 'diff', 'HEAD', '--binary', '--full-index'])
        assert hashlib.sha256(patch).hexdigest() == expected_patch_sha256
        status = output(base + ['git', '-C', repo, 'status', '--porcelain'])
        assert not any(line.startswith('??') for line in status.splitlines()), status
        if expected_patch_sha256 == EMPTY:
            assert not status
        return patch
    patch = verify_source()
    if patch:
        (folder / 'diagnostic-instrumentation.patch').write_bytes(patch)
    profile_dir = f'/task/outputs/profiles/{RUN}-{config}'
    launch = ['docker', 'exec', '-e', f'MOSS_CANDIDATE_SHA={HEAD}',
              '-e', f'SGLANG_DIFFUSION_TORCH_PROFILER_DIR={profile_dir}',
              container, 'sh', '-c',
              f'echo $$ > /task/block-runs/{RUN}-{config}/server.pid; exec sh /task/{script}']
    metadata = dict(config=config, arm=arm, source_sha=sha, source_diff_sha256=expected_patch_sha256, command=launch,
                    profile_dir=profile_dir, primary_benchmark=False,
                    native_pipeline_instrumentation=arm == 'omni',
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder / 'launch.sh').write_text((ROOT / script).read_text())
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.monotonic()
    with (folder / 'server.log').open('wb') as log:
        process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
    try:
        port, path = (31030, '/liveness') if arm == 'sgl' else (31031, '/health')
        while True:
            assert process.poll() is None, 'Diagnostic server exited.'
            if time.monotonic() - started > 600:
                raise TimeoutError('Diagnostic readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=2) as r:
                    ready = r.status == 200
            except Exception:
                ready = False
            if arm == 'sgl':
                ready = ready and b'Warmup requests: 100%' in (folder / 'server.log').read_bytes()
            if ready:
                break
            time.sleep(2)
        query = ('import json,pynvml as n; n.nvmlInit(); '
                 'print(json.dumps([[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(n.nvmlDeviceGetHandleByIndex(i))] for i in (0,1)]))')
        pids = json.loads(output(base + ['python', '-c', query]))
        assert pids[0] and not pids[1]
        metadata.update(launch_to_ready_seconds=time.monotonic() - started, gpu_container_pids=pids)
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

        def run(label, args):
            (folder / (label + '-command.json')).write_text(json.dumps(base + args, indent=2) + '\n')
            with (folder / (label + '.log')).open('wb') as log:
                subprocess.run(base + args, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps(dict(server=config, operation=label, status='passed')), flush=True)

        yield run, pids[0]
        verify_source()
    finally:
        pid_file = folder / 'server.pid'
        if process.poll() is None and pid_file.exists():
            subprocess.run(base + ['kill', '-TERM', pid_file.read_text().strip()], check=True)
        process.wait(timeout=60)
        for _ in range(30):
            if idle():
                break
            time.sleep(1)
        assert idle(), 'Diagnostic server did not release GPUs.'


def qualify(run, arm, label, quality, references, exact=False):
    run(label, ['python', '/task/submit_moss_qualification.py', arm, '--label', label, '--quality', quality])
    gate = ['python', '/task/compare_moss_qualification.py', f'{arm}-moss-{label}',
            '--root', '/task/outputs/qualification', '--fail-on-quality']
    for reference in references:
        gate += ['--reference', reference]
    if exact:
        gate += ['--require-pixel-identical']
    run(label + '-quality', gate)
    rows = json.loads((ROOT / 'outputs/qualification' / f'{arm}-moss-{label}' / 'results.json').read_text())
    assert rows[0]['sha256'] == rows[-1]['sha256']
    return rows[0]['sha256']


def main():
    assert_protocol_complete()
    main_lossless = f'sgl-moss-{RUN}-main-lossless'
    main_high_ref = 'sgl-moss-moss-qkv-final-paired1-00-main_cudnn'
    candidate_high_ref = 'sgl-moss-moss-qkv-final-paired1-01-candidate_cudnn'
    for config, repo, sha, script in [
        ('main', '/task/repos/sglang', BASE, 'launch_sgl_moss_cudnn.sh'),
        ('candidate', '/task/candidates/moss-qkv', HEAD, 'launch_sgl_moss_qkv_cudnn.sh'),
    ]:
        with server(config, 'sgl', repo, sha, script) as (run, _):
            label = f'{RUN}-{config}'
            refs = [main_high_ref] if config == 'main' else [main_lossless]
            qualify(run, 'sgl', label + '-lossless', 'lossless', refs, exact=config == 'candidate')
            high = qualify(run, 'sgl', label + '-high', 'high',
                           [main_high_ref if config == 'main' else candidate_high_ref], exact=True)
            if config == 'candidate':
                qualify(run, 'sgl', label + '-lossless-after-high', 'lossless', [main_lossless], exact=True)
            run('profile-high', ['python', '/task/submit_moss_qualification.py', 'sgl', '--label',
                                label + '-profile', '--quality', 'high', '--profile'])
            rows = json.loads((ROOT / 'outputs/qualification' / f'sgl-moss-{label}-profile' / 'results.json').read_text())
            assert rows[0]['sha256'] == high, 'Profile changed the actual generated PNG.'
    with server('omni-stage', 'omni', '/task/repos/vllm-omni', OMNI,
                'launch_omni_moss_stage_diagnostic.sh') as (run, pids):
        qualify(run, 'omni', RUN + '-omni-stage', 'high', [main_high_ref, candidate_high_ref])
        run('native-stage-aa', ['python', '/task/bench_moss_serving.py', 'omni', '--run-id', RUN + '-omni-stage',
                               '--repo', '/task/repos/vllm-omni', '--allowed-gpu-pids', *map(str, pids),
                               '--quality', 'high', '--configuration-label', 'native-pipeline-instrumented-diagnostic',
                               '--warmups', '4', '--repeats', '4'])
    print(json.dumps(dict(status='all_final_diagnostics_completed', primary_benchmark=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default=RUN)
    args = parser.parse_args()
    assert args.run_id.startswith('moss-qkv-final-diagnostics') and args.run_id.replace('-', '').isalnum()
    RUN = args.run_id
    main()
