"""Final-head request-quality transitions and observed BCG replay, never primary timing."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
RUN = 'qwen21-gate-up-v4-compatibility1'
CONTAINER = 'bbuf-diffusion-rtx6000-20261026-sgl'
BASE = 'af21e2b989e482cfdce39689126562f2209d53ac'
HEAD = 'efcb88a201a41b1fecc516a60df0ca0a34fb6dd5'
EMPTY = hashlib.sha256(b'').hexdigest()
OBSERVER = '592da6608fa449ccbda1721bfe4f8c1aa12729bace17afb158893e97ed3de588'
CASES = [f'{task}{size}' for size in (512,1024) for task in ('t2i','edit')]
STATUS = ROOT/(RUN+'-status.json')
state = dict(status='running', completed=[], primary_timing=False)


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])


def case_info(case):
    task = 't2i' if case.startswith('t2i') else 'edit'
    size = 512 if case.endswith('512') else 1024
    contract = ROOT/'inputs'/f'qwen21-base-{task}{"-512" if size==512 else ""}-workloads.json'
    high_reference = (f'sgl-qwen21-base-{task}-qwen21-base-512-qualification1-00-main-high'
                      if size==512 else f'sgl-qwen21-base-{task}-mainhigh1')
    return task, size, contract, high_reference


def label(case, config, phase, quality):
    return f'base-{case}-{RUN}-{config}-{phase}-{quality}'


def run(config):
    assert idle()
    repo = '/task/repos/sglang' if config=='main_eager' else '/task/candidates/qwen21-bf16-gate-up-v4'
    head = BASE if config=='main_eager' else HEAD
    launcher = {'main_eager':'launch_sgl_qwen21_main_compat.sh', 'final_eager':'launch_sgl_qwen21_v4_compat.sh', 'final_bcg':'launch_sgl_qwen21_v4_bcg_compat.sh'}[config]
    name = RUN+'-'+config
    folder = ROOT/'block-runs'/name
    folder.mkdir(exist_ok=False)
    assert command(['docker','exec',CONTAINER,'git','-C',repo,'rev-parse','HEAD'])==head
    patch = subprocess.check_output(['docker','exec',CONTAINER,'git','-C',repo,'diff','HEAD','--binary','--full-index'])
    assert hashlib.sha256(patch).hexdigest()==EMPTY
    assert hashlib.sha256((ROOT/'instrumentation/qwen21-bcg-observer/sitecustomize.py').read_bytes()).hexdigest()==OBSERVER
    launch = ['docker','exec','-e',f'SGL_REPO={repo}','-e',f'QWEN_CANDIDATE_DIFF_SHA={EMPTY}',
              '-e',f'QWEN_BCG_AUDIT_DIR=/task/outputs/diagnostics/{name}',
              '-e',f'SGLANG_DIFFUSION_TORCH_PROFILER_DIR=/task/outputs/profiles/{name}',
              CONTAINER,'sh','-c',f'echo $$ > /task/block-runs/{name}/server.pid; exec sh /task/{launcher}']
    meta = dict(source_sha=head, source_diff_sha256=EMPTY, observer_sha256=OBSERVER,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                launch_command=launch, primary_timing=False, requests=[])
    (folder/'launch.sh').write_bytes((ROOT/launcher).read_bytes())
    (folder/'source.patch').write_bytes(patch)
    def save():
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save()
    start = time.monotonic()
    with (folder/'server.log').open('xb') as stream:
        server = subprocess.Popen(launch,stdout=stream,stderr=subprocess.STDOUT)
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}')
            if time.monotonic()-start>1800:
                raise TimeoutError('readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:31050/liveness',timeout=2) as response:
                    ready = response.status==200
            except Exception:
                ready = False
            if ready and b'Warmup requests: 100%' in (folder/'server.log').read_bytes():
                break
            time.sleep(2)
        meta['launch_to_ready_seconds'] = time.monotonic()-start
        meta['gpu_processes_before_requests'] = command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])
        for case in CASES:
            task,size,contract,high_ref = case_info(case)
            phases = [('high_first','high'),('lossless','lossless'),('exact','exact')]
            if config!='main_eager':
                phases.append(('high_return','high'))
            phases.append(('profile','high'))
            for phase,quality in phases:
                state.update(active=config, case=case, phase=phase)
                STATUS.write_text(json.dumps(state,indent=2)+'\n')
                current_label = label(case,config,phase,quality)
                args = ['docker','exec',CONTAINER,'python','/task/submit_qwen21_compatibility.py','sgl',
                        '--contract','/task/inputs/'+contract.name,'--label',current_label,
                        '--quality',quality,'--repo',repo,'--expected-sha',head]
                all_prompts = config=='final_bcg' and phase=='high_first'
                if not all_prompts:
                    args += ['--prompt-indices','0']
                if phase=='profile':
                    args += ['--profile']
                entry = dict(case=case, phase=phase, quality=quality, command=args,
                             contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest())
                meta['requests'].append(entry)
                save()
                with (folder/f'{case}-{phase}.log').open('xb') as stream:
                    subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=7200)
                output = ROOT/'outputs/qualification'/('sgl-qwen21-'+current_label)
                rows = json.loads((output/'results.json').read_text())
                assert len(rows)==(4 if all_prompts else 1)
                reference = high_ref if quality=='high' else 'sgl-qwen21-'+label(case,'main_eager',phase,quality)
                references = json.loads((ROOT/'outputs/qualification'/reference/'results.json').read_text())
                assert len(references)>=len(rows)
                assert all(row['sha256']==ref['sha256'] for row,ref in zip(rows,references))
                entry.update(reference=reference, byte_exact_reference=True, output_count=len(rows))
                save()
        observed = [json.loads(line) for path in (ROOT/'outputs/diagnostics'/name).glob('*.jsonl') for line in path.read_text().splitlines()]
        by_id = {row['request_id']:row for row in observed if not row['is_warmup']}
        checks = []
        for entry in meta['requests']:
            output = ROOT/'outputs/qualification'/('sgl-qwen21-'+label(entry['case'],config,entry['phase'],entry['quality']))
            for directory in sorted(path for path in output.iterdir() if path.is_dir()):
                perf = json.loads((directory/'perf.json').read_text())
                record = by_id.pop(perf['request_id'])
                assert record['hook_sha256']==OBSERVER and record['capture_calls']==0
                task,size,_,_ = case_info(entry['case'])
                replay = 49 if config=='final_bcg' and entry['quality']=='high' and task=='t2i' else 0
                assert record['replay_calls']==replay, record
                assert record['python_model_forwards']==50-replay, record
                packed = (50-replay)*32 if config!='main_eager' and entry['quality']=='high' else 0
                assert sum(record['packed_python_calls'].values())==packed, record
                checks.append(dict(request_id=perf['request_id'],case=entry['case'],phase=entry['phase'],quality=entry['quality'],replay_calls=replay,packed_python_calls=packed))
        assert not by_id
        meta.update(status='complete', observed_request_checks=checks,
                    warmup_records=[row for row in observed if row['is_warmup']])
    except Exception as error:
        meta.update(status='failed',error=repr(error))
        raise
    finally:
        save()
        pid = folder/'server.pid'
        if server.poll() is None and pid.exists():
            subprocess.run(['docker','exec',CONTAINER,'kill','-TERM',pid.read_text().strip()],check=True)
        server.wait(timeout=60)
        for _ in range(30):
            if idle():
                break
            time.sleep(1)
        assert idle()


if __name__=='__main__':
    assert not STATUS.exists()
    plan = json.loads((ROOT/(RUN+'.json')).read_text())
    assert plan['head']==HEAD and plan['cases']==CASES
    for item in plan['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    try:
        for config in ['main_eager','final_eager','final_bcg']:
            run(config)
            state['completed'].append(config)
        state.update(status='raw_complete_audit_pending',active=None,case=None,phase=None)
    except Exception as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        STATUS.write_text(json.dumps(state,indent=2)+'\n')
