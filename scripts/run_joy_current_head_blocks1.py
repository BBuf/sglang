"""Run serialized, profiler-free same-GPU blocks from an explicit saved plan.

Run on the remote host, outside containers. Each block gets a fresh owned server.
The plan fixes ABBA order and repeats; this runner never chooses winners or drops rows.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def run_block(plan, index, config):
    spec = plan['configurations'][config]
    arm, script, repo, port = (spec[key] for key in ['arm', 'launcher', 'repo', 'port'])
    assert not plan.get('diagnostic_profiler') and not plan.get('profile_after_block')
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    name = f'{plan["id"]}-{index:02d}-{config}'
    folder = ROOT / 'block-runs' / name
    folder.mkdir(parents=True, exist_ok=False)
    assert idle(), 'GPU ownership preflight failed'
    candidate_source = plan.get('candidate_sources', {}).get(config, plan)
    head = command(['docker', 'exec', container, 'git', '-C', repo, 'rev-parse', 'HEAD'])
    assert head == spec['head']
    diff = subprocess.check_output(['docker', 'exec', container, 'git', '-C', repo,
                                    'diff', 'HEAD', '--binary', '--full-index'])
    assert not diff, 'Current-head supplement requires clean frozen source'
    (folder/'source.patch').write_bytes(diff)
    preflight_repo = repo if arm == 'sgl' else '/task/repos/sglang'
    with (folder/'preflight.log').open('xb') as stream:
        subprocess.run(['docker', 'exec', '-e', f'PYTHONPATH={preflight_repo}/python',
                        '-e', 'FLASHINFER_DISABLE_VERSION_CHECK=1',
                        'bbuf-diffusion-rtx6000-20261026-sgl', 'python3',
                        '/task/preflight_diffusion_current_head1.py', preflight_repo], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    contract = ROOT / Path(plan.get('contract', '/task/inputs/joy-workloads.json')).relative_to('/task')
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == plan['frozen_input_sha256']
    pid_path = f'/task/block-runs/{name}/server.pid'
    launch = ['docker', 'exec', '-e', f'SGL_REPO={repo}', container, 'sh', '-c',
              f'echo $$ > {shlex.quote(pid_path)}; exec sh /task/{script}']
    if config.startswith('candidate'):
        launch[2:2] = ['-e', 'JOY_CANDIDATE_SHA=' + candidate_source['candidate_head_sha']]
    metadata = dict(plan=plan, index=index, config=config, launch_command=launch,
                    source_sha=head, source_diff_sha256=hashlib.sha256(diff).hexdigest(),
                    stage_timing_synchronized=False, status='starting',
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (folder / 'launch.sh').write_text((ROOT / script).read_text())
    log_path = folder / 'server.log'
    start = time.monotonic()
    with log_path.open('xb') as stream:
        server = subprocess.Popen(launch, stdout=stream, stderr=subprocess.STDOUT)

    def native_gate():
        if arm == 'sgl':
            text = log_path.read_text(errors='replace')
            assert not any(marker in text for marker in [
                'Falling back to diffusers backend', 'Using diffusers backend',
                'Loaded diffusers pipeline']), 'Native backend gate failed'
    try:
        while True:
            native_gate()
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}: {log_path}')
            if time.monotonic()-start > 1200:
                raise TimeoutError('server readiness')
            path = '/liveness' if arm == 'sgl' else '/health'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=2) as response:
                    ready = response.status == 200
            except Exception:
                ready = False
            if arm == 'sgl':
                ready = ready and b'Warmup requests: 100%' in log_path.read_bytes()
            if ready:
                native_gate()
                break
            time.sleep(2)
        metadata['launch_to_ready_seconds'] = time.monotonic()-start
        pid = int((folder / 'server.pid').read_text())
        metadata['server_container_pid'] = pid
        nv_query = ('import json,pynvml as n; n.nvmlInit(); '
                    'print(json.dumps([[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(n.nvmlDeviceGetHandleByIndex(i))] for i in (0,1)]))')
        gpu_pids = json.loads(command(['docker','exec',container,'python','-c',nv_query]))
        assert gpu_pids[0] and not gpu_pids[1], gpu_pids
        metadata['gpu_container_pids'] = gpu_pids
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        if plan.get('qualify_each_block'):
            qual = ['docker','exec',container,'python','/task/submit_joy_candidate_qualification.py',arm,
                    '--label',name,'--quality',plan['quality']]
            if plan.get('contract'):
                qual += ['--contract', plan['contract']]
            metadata['qualification_command'] = qual
            with (folder/'qualification.log').open('wb') as stream:
                subprocess.run(qual, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if config in plan.get('quality_references', {}):
            gate = ['docker','exec',container,'python','/task/compare_moss_qualification.py',
                    f'{arm}-joy-{name}','--root','/task/outputs/qualification','--fail-on-quality']
            for ref in plan['quality_references'][config]:
                gate += ['--reference', ref]
            if config in plan.get('require_pixel_identical', []):
                gate += ['--require-pixel-identical']
            metadata['quality_gate_command'] = gate
            (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            with (folder/'quality-gate.log').open('wb') as stream:
                subprocess.run(gate, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if config in plan.get('byte_identical_quality_reference', {}):
            current = ROOT/'outputs/qualification'/f'{arm}-joy-{name}'/'results.json'
            reference = ROOT/'outputs/qualification'/plan['byte_identical_quality_reference'][config]/'results.json'
            assert [r['sha256'] for r in json.loads(current.read_text())] == [
                r['sha256'] for r in json.loads(reference.read_text())], 'Quality outputs changed from frozen same-source reference'
        bench = ['docker','exec',container,'python','/task/bench_joy_serving.py',arm,
                 '--run-id',name,'--repo',repo,'--allowed-gpu-pids',*map(str,gpu_pids[0]),
                 '--quality',plan['quality'],'--configuration-label',plan['configuration_labels'][config],
                 '--warmups',str(plan['warmups']),'--repeats',str(plan['repeats'])]
        if plan.get('diagnostic_profiler'):
            bench += ['--diagnostic-profiler']
        if config.startswith('candidate'):
            bench += ['--expected-diff-sha256',candidate_source['candidate_diff_sha256']]
            bench += ['--expected-sha',candidate_source['candidate_head_sha']]
        if plan.get('contract'):
            bench += ['--contract', plan['contract']]
        metadata['benchmark_command'] = bench
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        with (folder/'benchmark.log').open('wb') as stream:
            subprocess.run(bench, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((ROOT/'outputs/benchmarks'/name/'summary.json').read_text())
        if arm == 'sgl':
            assert summary['all_outputs_byte_identical'], 'Fixed-seed SGL outputs changed inside this process'
        if plan.get('qualify_each_block'):
            reference = json.loads((ROOT/'outputs/qualification'/f'{arm}-joy-{name}'/'results.json').read_text())[0]['sha256']
            measured = json.loads((ROOT/'outputs/benchmarks'/name/'results.json').read_text())
            if arm == 'sgl':
                assert all(row['sha256'] == reference for row in measured), 'Benchmark differs from qualified short output'
            else:
                quality = ['docker', 'exec', container, 'python', '/task/check_joy_benchmark_quality.py', name, '--reference', plan['omni_short_quality_reference']]
                with (folder/'benchmark-quality.log').open('wb') as stream:
                    subprocess.run(quality, stdout=stream, stderr=subprocess.STDOUT, check=True)
        native_gate()
        metadata['status'] = 'complete'
        metadata['capture_lines'] = [line for line in log_path.read_text(errors='replace').splitlines() if '[Diffusion BCG]' in line]
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(dict(block=name,status='completed',summary=summary)),flush=True)
    except Exception as exc:
        metadata.update(status='failed', error=repr(exc))
        (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        raise
    finally:
        pid_file = folder/'server.pid'
        if server.poll() is None and pid_file.exists():
            # Only signal the PID started and recorded by this block in its own container.
            subprocess.run(['docker','exec',container,'kill','-TERM',pid_file.read_text().strip()],check=True)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Owned server did not stop; do not start the next block')
        for _ in range(30):
            if idle():
                break
            time.sleep(1)
        assert idle(), 'GPU processes remain; next block prohibited'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    status_path = ROOT/(plan['id']+'-status.json')
    assert not status_path.exists(), 'No automatic retry or output overwrite'
    assert plan['qualify_each_block'] and plan['quality'] == 'high'
    assert plan['SGLANG_DIFFUSION_SYNC_STAGE_PROFILING'] == 0
    for path, expected in plan['sha256'].items():
        source = ROOT/(Path(path).name if path.startswith('scripts/') else path)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, path
    for path in plan['required_admissions']:
        assert json.loads((ROOT/path).read_text())['passed'], path
    archive = json.loads((ROOT/'evidence/checkpoints/sana-video-depthwise-final-primary1-local-verification.json').read_text())
    assert archive['all_member_hashes_verified'] and archive['all_members_extracted']
    state = dict(status='running', completed=[], primary_stage_timings_valid=False)
    try:
        for index, config in enumerate(plan['order']):
            state.update(active_index=index, active_configuration=config)
            status_path.write_text(json.dumps(state, indent=2)+'\n')
            run_block(plan, index, config)
            state['completed'].append(f'{plan["id"]}-{index:02d}-{config}')
        state.update(status='raw_complete_audit_pending', active_index=None, active_configuration=None)
    except Exception as exc:
        state.update(status='failed_preserve_outputs_no_retry', error=repr(exc))
        raise
    finally:
        status_path.write_text(json.dumps(state, indent=2)+'\n')
