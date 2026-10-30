"""Revalidate current Joy quality switching and collect separate synchronized traces."""
import hashlib
import json
import shlex
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

from analyze_joy_gelu_profiles import analyze

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
CONTAINER = 'bbuf-diffusion-rtx6000-20261026-sgl'
PLAN_PATH = ROOT/'joy-input-lifetime-repair1-diagnostics1.json'
PLAN = json.loads(PLAN_PATH.read_text())
STATUS_PATH = ROOT/(PLAN['id']+'-status.json')
assert not STATUS_PATH.exists(), 'No automatic retry or overwrite'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


for path, expected in PLAN['sha256'].items():
    actual = ROOT/(Path(path).name if path.startswith('scripts/') else path)
    assert digest(actual) == expected, path
for path in PLAN['required_admissions']:
    assert read(ROOT/path)['passed'], path
for shape in ('square', 'wide'):
    receipt = read(ROOT/'evidence/checkpoints'/f'joy-input-lifetime-repair1-{shape}-primary1-local-verification.json')
    assert receipt['all_member_hashes_verified'] and receipt['all_members_extracted']
state = dict(status='running', primary_timing=False, completed=[], profiles={}, outputs=0)


def save():
    STATUS_PATH.write_text(json.dumps(state, indent=2)+'\n')


def run(shape, source):
    spec = PLAN['sources'][source]
    contract = PLAN['contracts'][shape]
    name = f'{PLAN["id"]}-{shape}-{source}'
    folder = ROOT/'block-runs'/name
    folder.mkdir(exist_ok=False)
    assert idle(), 'A GPU process is present'
    assert command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()[0] == PLAN['physical_gpu_uuid']
    assert command(['docker', 'exec', CONTAINER, 'git', '-C', spec['repo'], 'rev-parse', 'HEAD']) == spec['head']
    diff = subprocess.check_output(['docker', 'exec', CONTAINER, 'git', '-C', spec['repo'], 'diff', 'HEAD', '--binary', '--full-index'])
    assert not diff, 'Source changed'
    (folder/'source.patch').write_bytes(diff)
    with (folder/'preflight.log').open('xb') as stream:
        subprocess.run(['docker', 'exec', '-e', 'PYTHONPATH='+spec['repo']+'/python',
                        '-e', 'FLASHINFER_DISABLE_VERSION_CHECK=1', CONTAINER, 'python',
                        '/task/preflight_diffusion_current_head1.py', spec['repo']],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    profile_dir = '/task/outputs/profiles/'+name
    pid_path = '/task/block-runs/'+name+'/server.pid'
    launch = ['docker', 'exec', '-e', 'SGL_REPO='+spec['repo'],
              '-e', 'JOY_CANDIDATE_SHA='+spec['head'],
              '-e', 'SGLANG_DIFFUSION_TORCH_PROFILER_DIR='+profile_dir,
              CONTAINER, 'sh', '-c',
              f'echo $$ > {shlex.quote(pid_path)}; exec sh /task/{PLAN["launcher"]}']
    metadata = dict(source=source, shape=shape, source_sha=spec['head'], source_diff_sha256=digest(folder/'source.patch'),
                    plan_sha256=digest(PLAN_PATH), primary_timing=False, synchronized_stages=True,
                    launch_command=launch, qualifications=[], status='starting')
    (folder/'launch.sh').write_bytes((ROOT/PLAN['launcher']).read_bytes())
    log = folder/'server.log'
    start = time.monotonic()
    with log.open('xb') as stream:
        server = subprocess.Popen(launch, stdout=stream, stderr=subprocess.STDOUT)

    def native_gate():
        contents = log.read_text(errors='replace')
        assert not any(marker in contents for marker in PLAN['forbidden_backend_markers']), 'Native backend gate failed'

    try:
        while True:
            native_gate()
            assert server.poll() is None, 'Server exited'
            if time.monotonic()-start > 1200:
                raise TimeoutError('Server readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:31040/liveness', timeout=2) as response:
                    ready = response.status == 200
            except Exception:
                ready = False
            if ready and b'Warmup requests: 100%' in log.read_bytes():
                break
            time.sleep(2)
        metadata.update(launch_to_ready_seconds=time.monotonic()-start,
                        server_container_pid=int((folder/'server.pid').read_text()))
        phases = [(str(i), quality, False) for i, quality in enumerate(PLAN['quality_order'])]
        if shape in PLAN['profile_shapes']:
            phases.append(('profile', 'high', True))
        for phase, quality, profiled in phases:
            label = name+'-'+phase+'-'+quality
            args = ['docker', 'exec', CONTAINER, 'python', '/task/submit_joy_candidate_qualification.py',
                    'sgl', '--label', label, '--contract', contract, '--quality', quality]
            if profiled:
                args.append('--profile')
            with (folder/f'{phase}-{quality}.log').open('xb') as stream:
                subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=True)
            native_gate()
            output = ROOT/'outputs/qualification'/('sgl-joy-'+label)
            rows = read(output/'results.json')
            reference_name = PLAN['references'][shape][source][quality]
            reference = ROOT/'outputs/qualification'/reference_name
            expected = read(reference/'results.json')
            assert len(rows) == (1 if profiled else 4)
            assert [row['sha256'] for row in rows] == [row['sha256'] for row in expected[:len(rows)]]
            stages = []
            for row, old in zip(rows, expected):
                request = output/Path(row['request']).name
                assert digest(request/'output.png') == row['sha256']
                assert digest(reference/Path(old['request']).name/'output.png') == old['sha256']
                assert row['profiled'] == profiled and not row['benchmark']
                if not profiled:
                    perf = read(request/'perf.json')
                    duration = next(step['duration_ms']/1000 for step in perf['steps'] if step['name'] == 'DenoisingStage')
                    assert 0 < duration < row['http_seconds']
                    stages.append(dict(request=request.name, denoise_seconds=duration, http_seconds=row['http_seconds']))
            item = dict(phase=phase, quality=quality, profiled=profiled, command=args,
                        reference=reference_name, outputs=len(rows), byte_identical_to_reference=True,
                        stages=stages, diagnostic_only=True)
            metadata['qualifications'].append(item)
            state['outputs'] += len(rows)
            (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
            save()
        traces = list((ROOT/'outputs/profiles'/name).glob('*.trace.json.gz'))
        assert len(traces) == (1 if shape in PLAN['profile_shapes'] else 0)
        if traces:
            analysis = analyze(traces[0], 3)
            assert analysis['cuda_graph_launch_count'] == 0
            if source == 'current':
                assert analysis['fused_custom_op_calls'] == analysis['activation_calls'] == 80
            else:
                assert analysis['fused_custom_op_calls'] == analysis['activation_calls'] == 0
            (folder/'profile-analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
            state['profiles'][shape+'-'+source] = dict(path=str(traces[0].relative_to(ROOT)),
                sha256=digest(traces[0]), bytes=traces[0].stat().st_size,
                gelu_cpu_calls=analysis['gelu_cpu_calls'], fused_custom_op_calls=analysis['fused_custom_op_calls'],
                activation_calls=analysis['activation_calls'], kernel_count=analysis['kernel_count'],
                kernel_sum_ms=analysis['kernel_sum_ms'])
        high = next(item for item in metadata['qualifications'] if item['quality'] == 'high' and not item['profiled'])
        metadata['high_short_denoise_p50_seconds'] = statistics.median(high['stages'][i]['denoise_seconds'] for i in (0, 3))
        metadata['status'] = 'raw_quality_and_diagnostics_complete_local_audit_pending'
    except Exception as exc:
        metadata.update(status='failed_preserve_outputs_no_retry', error=repr(exc))
        raise
    finally:
        (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        pid = folder/'server.pid'
        if server.poll() is None and pid.exists():
            subprocess.run(['docker', 'exec', CONTAINER, 'kill', '-TERM', pid.read_text().strip()], check=True)
        server.wait(timeout=60)
        for _ in range(30):
            if idle():
                break
            time.sleep(1)
        assert idle(), 'Owned process remains; no next run'


try:
    for shape in PLAN['shapes']:
        for source in PLAN['source_order']:
            state.update(active_shape=shape, active_source=source)
            save()
            run(shape, source)
            state['completed'].append(shape+'-'+source)
        if shape in PLAN['profile_shapes']:
            assert state['profiles'][shape+'-main']['gelu_cpu_calls']-state['profiles'][shape+'-current']['gelu_cpu_calls'] == 80
    assert state['outputs'] == PLAN['expected_outputs']
    state.update(status='raw_complete_archive_and_local_audit_pending', active_shape=None, active_source=None)
except Exception as exc:
    state.update(status='failed_preserve_outputs_no_retry', error=repr(exc))
    raise
finally:
    save()
