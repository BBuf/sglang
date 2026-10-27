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
CONFIGS = {
    'main': ('sgl', 'launch_sgl_joy_candidate.sh', '/task/repos/sglang', 31040),
    'candidate': ('sgl', 'launch_sgl_joy_candidate.sh', '/task/candidates/joy-image-gelu', 31040),
    'candidate_precleanup': ('sgl', 'launch_sgl_joy_candidate.sh', '/task/candidates/joy-image-gelu-precleanup', 31040),
    'omni': ('omni', 'launch_omni_joy_qualification.sh', '/task/repos/vllm-omni', 31041),
    'omni_compile': ('omni', 'launch_omni_joy_compile.sh', '/task/repos/vllm-omni', 31041),
    'omni_compile_casts': ('omni', 'launch_omni_joy_compile_casts.sh', '/task/repos/vllm-omni', 31041),
    'omni_compile_profile': ('omni', 'launch_omni_joy_compile_profile.sh', '/task/repos/vllm-omni', 31041),
}


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def run_block(plan, index, config):
    arm, script, repo, port = CONFIGS[config]
    assert plan.get('diagnostic_profiler', False) == config.endswith('_profile')
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    name = f'{plan["id"]}-{index:02d}-{config}'
    folder = ROOT / 'block-runs' / name
    folder.mkdir(parents=True, exist_ok=False)
    assert idle(), 'GPU ownership preflight failed'
    candidate_source = plan.get('candidate_sources', {}).get(config, plan)
    if config.startswith('candidate'):
        head = command(['docker', 'exec', container, 'git', '-C', repo, 'rev-parse', 'HEAD'])
        assert head == candidate_source['candidate_head_sha']
        diff = subprocess.check_output(['docker', 'exec', container, 'git', '-C', repo,
                                        'diff', 'HEAD', '--binary', '--full-index'])
        assert hashlib.sha256(diff).hexdigest() == candidate_source['candidate_diff_sha256']
    contract = ROOT / Path(plan.get('contract', '/task/inputs/joy-workloads.json')).relative_to('/task')
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == plan['frozen_input_sha256']
    pid_path = f'/task/block-runs/{name}/server.pid'
    launch = ['docker', 'exec', '-e', f'SGL_REPO={repo}', container, 'sh', '-c',
              f'echo $$ > {shlex.quote(pid_path)}; exec sh /task/{script}']
    if config.startswith('candidate'):
        launch[2:2] = ['-e', 'JOY_CANDIDATE_SHA=' + candidate_source['candidate_head_sha']]
    metadata = dict(plan=plan, index=index, config=config, launch_command=launch,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (folder / 'launch.sh').write_text((ROOT / script).read_text())
    log_path = folder / 'server.log'
    start = time.monotonic()
    server = subprocess.Popen(launch, stdout=log_path.open('wb'), stderr=subprocess.STDOUT)
    try:
        while True:
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
        if plan.get('profile_after_block') and arm == 'sgl':
            profile = ['docker','exec',container,'python','/task/submit_joy_candidate_qualification.py',
                       arm,'--label',name+'-profile','--quality',plan['quality'],'--profile']
            if plan.get('contract'):
                profile += ['--contract', plan['contract']]
            with (folder/'profile.log').open('wb') as stream:
                subprocess.run(profile,stdout=stream,stderr=subprocess.STDOUT,check=True)
        metadata['capture_lines'] = [line for line in log_path.read_text(errors='replace').splitlines() if '[Diffusion BCG]' in line]
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(dict(block=name,status='completed',summary=summary)),flush=True)
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
    p=argparse.ArgumentParser()
    p.add_argument('plan',type=Path)
    args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    for i, config in enumerate(plan['order']):
        run_block(plan,i,config)
