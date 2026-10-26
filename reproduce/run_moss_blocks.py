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
    'main': ('sgl', 'launch_sgl_moss_qualification.sh', '/task/repos/sglang', 31030),
    'candidate': ('sgl', 'launch_sgl_moss_qkv_initial.sh', '/task/candidates/moss-qkv', 31030),
    'main_cudnn': ('sgl', 'launch_sgl_moss_cudnn.sh', '/task/repos/sglang', 31030),
    'candidate_cudnn': ('sgl', 'launch_sgl_moss_qkv_cudnn.sh', '/task/candidates/moss-qkv', 31030),
    'omni': ('omni', 'launch_omni_moss_compile_casts.sh', '/task/repos/vllm-omni', 31031),
}


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def run_block(plan, index, config):
    arm, script, repo, port = CONFIGS[config]
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    name = f'{plan["id"]}-{index:02d}-{config}'
    folder = ROOT / 'block-runs' / name
    folder.mkdir(parents=True, exist_ok=False)
    assert idle(), 'GPU ownership preflight failed'
    pid_path = f'/task/block-runs/{name}/server.pid'
    launch = ['docker', 'exec', container, 'sh', '-c',
              f'echo $$ > {shlex.quote(pid_path)}; exec sh /task/{script}']
    if config.startswith('candidate') and plan.get('candidate_head_sha'):
        launch[2:2] = ['-e', 'MOSS_CANDIDATE_SHA=' + plan['candidate_head_sha']]
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
            if time.monotonic()-start > 600:
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
            qual = ['docker','exec',container,'python','/task/submit_moss_qualification.py',arm,
                    '--label',name,'--quality',plan['quality']]
            metadata['qualification_command'] = qual
            with (folder/'qualification.log').open('wb') as stream:
                subprocess.run(qual, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if config in plan.get('quality_references', {}):
            gate = ['docker','exec',container,'python','/task/compare_moss_qualification.py',
                    f'{arm}-moss-{name}','--root','/task/outputs/qualification','--fail-on-quality']
            for ref in plan['quality_references'][config]:
                gate += ['--reference', ref]
            if config in plan.get('require_pixel_identical', []):
                gate += ['--require-pixel-identical']
            metadata['quality_gate_command'] = gate
            (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            with (folder/'quality-gate.log').open('wb') as stream:
                subprocess.run(gate, stdout=stream, stderr=subprocess.STDOUT, check=True)
        bench = ['docker','exec',container,'python','/task/bench_moss_serving.py',arm,
                 '--run-id',name,'--repo',repo,'--allowed-gpu-pids',*map(str,gpu_pids[0]),
                 '--quality',plan['quality'],'--configuration-label',plan['configuration_labels'][config],
                 '--warmups',str(plan['warmups']),'--repeats',str(plan['repeats'])]
        if config.startswith('candidate'):
            bench += ['--expected-diff-sha256',plan['candidate_diff_sha256']]
            if plan.get('candidate_head_sha'):
                bench += ['--expected-sha',plan['candidate_head_sha']]
        metadata['benchmark_command'] = bench
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        with (folder/'benchmark.log').open('wb') as stream:
            subprocess.run(bench, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((ROOT/'outputs/benchmarks'/name/'summary.json').read_text())
        assert summary['all_outputs_byte_identical'], 'Fixed-seed outputs changed inside this process'
        if plan.get('qualify_each_block'):
            reference = json.loads((ROOT/'outputs/qualification'/f'{arm}-moss-{name}'/'results.json').read_text())[0]['sha256']
            measured = json.loads((ROOT/'outputs/benchmarks'/name/'results.json').read_text())
            assert all(row['sha256'] == reference for row in measured), 'Benchmark differs from qualified short output'
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
