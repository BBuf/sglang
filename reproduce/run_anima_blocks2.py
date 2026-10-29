"""Pinned serial Anima qualification, optional profiles and admitted prompt timings."""
import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path
ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
BASE = 'af21e2b989e482cfdce39689126562f2209d53ac'
OMNI = '817d4c17feb702289a2d2b2cbacfd594c262b7a7'
EMPTY = hashlib.sha256(b'').hexdigest()

def command(args):
    return subprocess.check_output(args, text=True).strip()

def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])

def run_block(plan, index, config):
    spec = plan['configurations'][config]
    arm, repo = spec['arm'], spec['repo']
    sha = spec.get('head', BASE if arm == 'sgl' else OMNI)
    diffsha = spec.get('diff_sha256', EMPTY)
    name = f'{plan["id"]}-{index:02d}-{config}'
    folder = ROOT/'block-runs'/name
    folder.mkdir(exist_ok=False)
    container = 'bbuf-diffusion-rtx6000-20261026-' + arm
    port = 31050 if arm == 'sgl' else 31051
    assert idle(), 'Unowned GPU work present'
    with (folder/'preflight.log').open('xb') as stream:
        subprocess.run(['docker','exec','-e','PYTHONPATH=/task/repos/sglang/python',
                        'bbuf-diffusion-rtx6000-20261026-sgl','python3','/task/preflight_ming_environment.py'],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert command(['docker','exec',container,'git','-C',repo,'rev-parse','HEAD']) == sha
    patch = subprocess.check_output(['docker','exec',container,'git','-C',repo,'diff','HEAD','--binary','--full-index'])
    assert hashlib.sha256(patch).hexdigest() == diffsha
    (folder/'source.patch').write_bytes(patch)
    script = spec['launcher']
    script_bytes = (ROOT/script).read_bytes()
    assert hashlib.sha256(script_bytes).hexdigest() == plan['sha256']['scripts/'+script]
    (folder/'launch.sh').write_bytes(script_bytes)
    launch = ['docker','exec','-e',f'SGL_REPO={repo}','-e',f'SGL_EXPECTED_HEAD={sha}',
              '-e',f'SGLANG_DIFFUSION_TORCH_PROFILER_DIR=/task/outputs/profiles/{name}',
              container,'sh','-c',f'echo $$ > /task/block-runs/{name}/server.pid; exec sh /task/{script}']
    meta = dict(plan=plan,index=index,config=config,source_sha=sha,source_diff_sha256=diffsha,
                launch_command=launch,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),commands=[])
    def save():
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    def execute(args,label):
        meta['commands'].append(args)
        save()
        with (folder/(label+'.log')).open('xb') as stream:
            subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=7200)
    save()
    start = time.monotonic()
    log = folder/'server.log'
    with log.open('xb') as stream:
        server = subprocess.Popen(launch,stdout=stream,stderr=subprocess.STDOUT)
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exit {server.returncode}: {log}')
            if time.monotonic()-start > 1800:
                raise TimeoutError('readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}'+('/liveness' if arm=='sgl' else '/health'),timeout=2) as response:
                    ready = response.status == 200
            except Exception:
                ready = False
            if arm == 'sgl':
                ready = ready and b'Warmup requests: 100%' in log.read_bytes()
            if ready:
                break
            time.sleep(2)
        meta['launch_to_ready_seconds'] = time.monotonic()-start
        q = 'import json,pynvml as n;n.nvmlInit();print(json.dumps([[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(n.nvmlDeviceGetHandleByIndex(i))] for i in (0,1)]))'
        pids = json.loads(command(['docker','exec',container,'python3','-c',q]))
        assert pids[0] and not pids[1]
        meta['gpu_container_pids'] = pids
        contract = ROOT/plan['contract']
        assert hashlib.sha256(contract.read_bytes()).hexdigest() == plan['sha256'][plan['contract']]
        label = name + '-' + plan['quality']
        qualified = f'{arm}-anima-{label}'
        common = ['docker','exec',container,'python3','/task/submit_anima_candidate.py',arm,
                  '--quality',plan['quality'],'--contract','/task/'+plan['contract'],
                  '--source-root',repo,'--source-sha',sha,'--diff-sha256',diffsha]
        execute(common+['--label',label], 'qualification')
        refs = spec.get('references', [])
        if refs:
            cmd = ['docker','exec',container,'python3','/task/compare_moss_qualification.py',qualified,
                   '--root','/task/outputs/qualification']
            for ref in refs:
                cmd += ['--reference',ref]
            execute(cmd,'quality')
        if spec.get('exact_reference'):
            comparisons = json.loads((ROOT/'outputs/qualification'/(qualified+'-quality.json')).read_text())['results']
            exact_rows = [row for row in comparisons if row['reference']==spec['exact_reference']]
            assert len(exact_rows)==4 and all(row['pixel_identical'] for row in exact_rows), 'Candidate changed any frozen main output'
        if plan.get('benchmark',False):
            comparisons = json.loads((ROOT/'outputs/qualification'/(qualified+'-quality.json')).read_text())['results']
            for prompt in plan['measured_prompts']:
                matched = [r for r in comparisons if r['prompt'].endswith('-'+prompt)]
                assert matched and all(r['passes_cross_framework_gate'] for r in matched), 'Invalid primary timing cell'
                if spec.get('exact_reference'):
                    assert all(r['pixel_identical'] for r in matched if r['reference']==spec['exact_reference'])
                runid = name+'-'+prompt
                execute(['docker','exec',container,'python3','/task/bench_anima_serving.py',arm,
                         '--run-id',runid,'--repo',repo,'--allowed-gpu-pids',*map(str,pids[0]),
                         '--quality',plan['quality'],'--configuration-label',config,'--warmups',str(plan['warmups']),
                         '--repeats',str(plan['repeats']),'--expected-sha',sha,'--expected-diff-sha256',diffsha,
                         '--contract','/task/'+plan['contract'],'--prompt-id',prompt],prompt+'-benchmark')
                actual = json.loads((ROOT/'outputs/benchmarks'/runid/'results.json').read_text())
                qrows = json.loads((ROOT/'outputs/qualification'/qualified/'results.json').read_text())
                reference = next(r['sha256'] for r in qrows if Path(r['request']).name.endswith('-'+prompt))
                assert all(r['sha256']==reference for r in actual), 'Measured output changed'
        for quality in spec.get('extra_qualities', []):
            extra = common.copy()
            extra[extra.index('--quality')+1] = quality
            extra_label = label+'-switch-'+quality
            execute(extra+['--label',extra_label], 'quality-switch-'+quality)
            extra_folder = ROOT/'outputs/qualification'/f'{arm}-anima-{extra_label}'
            for image in extra_folder.glob('*/output.png'):
                assert image.read_bytes() == (ROOT/'outputs/qualification'/qualified/image.relative_to(extra_folder)).read_bytes(), 'Quality switch changed exact Anima output'
        if spec.get('profile'):
            execute(common+['--label',label+'-profile','--profile'],'profile')
        if arm == 'sgl':
            assert not any(m in log.read_text() for m in ['Falling back to diffusers backend','Using diffusers backend','Loaded diffusers pipeline'])
        meta['status'] = 'complete'
    except Exception as exc:
        meta.update(status='failed',error=repr(exc))
        raise
    finally:
        save()
        pid = folder/'server.pid'
        if server.poll() is None and pid.exists():
            subprocess.run(['docker','exec',container,'kill','-TERM',pid.read_text().strip()],check=True)
        server.wait(timeout=60)
        for _ in range(30):
            if idle():
                break
            time.sleep(1)
        assert idle(), 'GPU work remains; next block prohibited'
    return name

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('plan',type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    for path,sha in plan['sha256'].items():
        actual = ROOT/(Path(path).name if path.startswith('scripts/') else path)
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == sha, path
    status = ROOT/(plan['id']+'-status.json')
    assert not status.exists()
    state = dict(status='running',completed=[])
    try:
        for i,config in enumerate(plan['order']):
            state.update(active_index=i,active_configuration=config)
            status.write_text(json.dumps(state,indent=2)+'\n')
            state['completed'].append(run_block(plan,i,config))
        state.update(status='raw_complete_audit_pending',active_index=None,active_configuration=None)
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        status.write_text(json.dumps(state,indent=2)+'\n')
