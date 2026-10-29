"""Separate final-source native quality and BCG lifecycle diagnostics, serial GPU0."""
import hashlib
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
RUN = 'sana-video-depthwise-final-compatibility1'
CONTAINER = 'bbuf-diffusion-rtx6000-20261026-sgl'
STATUS = ROOT/(RUN+'-status.json')
PLAN = json.loads((ROOT/(RUN+'-plan.json')).read_text())
completed = []


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def save(status, **fields):
    STATUS.write_text(json.dumps(dict(status=status, benchmark=False, completed=completed, **fields), indent=2)+'\n')


def run(index, configuration):
    spec = PLAN['configurations'][configuration]
    assert idle(), 'Existing GPU process prohibits diagnostic launch'
    folder = ROOT/'block-runs'/f'{RUN}-{index:02d}-{configuration}'
    folder.mkdir(exist_ok=False)
    preflight = ['docker', 'exec', '-e', 'PYTHONPATH=/task/repos/sglang/python', CONTAINER, 'python3', '/task/preflight_ming_environment.py']
    with (folder/'preflight.log').open('xb') as stream:
        subprocess.run(preflight, stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert command(['docker', 'exec', CONTAINER, 'git', '-C', spec['repo'], 'rev-parse', 'HEAD']) == spec['head']
    patch = subprocess.check_output(['docker', 'exec', CONTAINER, 'git', '-C', spec['repo'], 'diff', 'HEAD', '--binary', '--full-index'])
    assert not patch
    name = folder.name
    launch = ['docker', 'exec', '-e', f'SANA_VIDEO_BCG_AUDIT_DIR=/task/outputs/diagnostics/{name}',
              '-e', f'SGLANG_DIFFUSION_TORCH_PROFILER_DIR=/task/outputs/profiles/{name}', CONTAINER,
              'sh', '-c', f'echo $$ > /task/block-runs/{name}/server.pid; exec sh /task/{spec["launcher"]}']
    (folder/'launch.sh').write_bytes((ROOT/spec['launcher']).read_bytes())
    metadata = dict(configuration=configuration, source_sha=spec['head'], source_diff_sha256=hashlib.sha256(patch).hexdigest(),
                    physical_gpus=[0], physical_gpu_pool=[0,1], launch_command=launch, benchmark=False, commands=[],
                    observer_sha256=PLAN['sha256']['scripts/sana_video_depthwise_bcg_sitecustomize1.py'],
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    samples, errors = [], []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append(dict(monotonic=time.monotonic(), csv=command(['nvidia-smi', '--query-gpu=uuid,memory.used,power.draw,temperature.gpu,clocks.sm,utilization.gpu', '--format=csv,noheader'])))
            except Exception as exc:
                errors.append(repr(exc))
                return
            stop.wait(1)

    thread = threading.Thread(target=monitor)
    thread.start()
    log = folder/'server.log'
    started = time.monotonic()
    save('starting', active_configuration=configuration)
    with log.open('xb') as stream:
        server = subprocess.Popen(launch, stdout=stream, stderr=subprocess.STDOUT)
    try:
        while True:
            assert server.poll() is None, f'Server exited; see {log}'
            assert time.monotonic()-started < 3600, 'Readiness timeout'
            try:
                with urllib.request.urlopen('http://127.0.0.1:31050/liveness', timeout=2) as response:
                    ready = response.status == 200
            except Exception:
                ready = False
            if ready and b'Warmup requests: 100%' in log.read_bytes():
                break
            time.sleep(2)
        metadata['launch_to_ready_seconds'] = time.monotonic()-started
        metadata['gpu_snapshot_before_requests'] = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'])
        metadata['server_container_pid'] = int((folder/'server.pid').read_text())
        for profiled in ([False, True] if configuration in PLAN['profile_configurations'] else [False]):
            label = name+('-profile' if profiled else '')
            cmd = ['docker', 'exec', CONTAINER, 'python3', '/task/submit_sana_video_compatibility1.py', 'sgl',
                   '--repo', spec['repo'], '--source-sha', spec['head'], '--expected-diff-sha256', hashlib.sha256(patch).hexdigest(),
                   '--label', label, '--contract', '/task/inputs/sana-video-480p-t2v-workloads.json']
            if profiled:
                cmd.append('--profile')
            else:
                cmd.extend(['--sequence-json', '/task/sana-video-final-quality-sequence1.json'])
            metadata['commands'].append(cmd)
            (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
            save('profiling' if profiled else 'qualifying', active_configuration=configuration, label=label)
            with (folder/('profile.log' if profiled else 'requests.log')).open('xb') as stream:
                subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
        text = log.read_text()
        assert not any(marker in text for marker in ['Falling back to diffusers backend', 'Using diffusers backend', 'Loaded diffusers pipeline'])
        metadata['status'] = 'complete_raw_audit_pending'
        assert not errors, errors
    except Exception as exc:
        metadata.update(status='failed_preserve_outputs', error=repr(exc))
        raise
    finally:
        if server.poll() is None and (folder/'server.pid').exists():
            subprocess.run(['docker', 'exec', CONTAINER, 'kill', '-TERM', (folder/'server.pid').read_text().strip()], check=True)
        server.wait(timeout=60)
        stop.set()
        thread.join()
        (folder/'gpu-samples.csv.json').write_text(json.dumps(samples)+'\n')
        metadata['gpu_monitor_errors'] = errors
        (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        for _ in range(30):
            if idle():
                break
            time.sleep(1)
        assert idle(), 'Remaining GPU processes prohibit another arm'
    completed.append(name)


if __name__ == '__main__':
    assert not STATUS.exists(), 'No automatic retry or output overwrite'
    try:
        assert json.loads((ROOT/'evidence/preflight/sana-video-depthwise-v1-paired1-admission.json').read_text())['passed']
        tests = json.loads((ROOT/'block-runs/sana-video-depthwise-final-tests1/results.json').read_text())
        assert len(tests)==3 and all(row['returncode']==0 for row in tests)
        assert all(row['source_sha']==PLAN['final_head'] for row in tests)
        for path, expected in PLAN['sha256'].items():
            actual = ROOT/(Path(path).name if path.startswith('scripts/') or path.startswith('evidence/plans/') else path)
            assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected, path
        assert hashlib.sha256((ROOT/'sana-video-depthwise-bcg-observer1/sitecustomize.py').read_bytes()).hexdigest() == PLAN['sha256']['scripts/sana_video_depthwise_bcg_sitecustomize1.py']
        for index, configuration in enumerate(PLAN['order']):
            run(index, configuration)
        save('raw_complete_quality_replay_and_profile_audit_pending', active_configuration=None)
    except Exception as exc:
        save('failed_preserve_all_outputs_no_retry', error=repr(exc))
        raise
