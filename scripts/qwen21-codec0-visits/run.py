"""Qwen-Image 2.1 native image admission; no performance conclusions before quality validation."""
import datetime
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path('/data/xyzhang/diffusion-6000-20261113')
arm,run,plan=sys.argv[1:]
assert arm in ('sgl','omni') and Path(run).name==run
plan_path=(ROOT/plan).resolve()
assert plan_path.is_relative_to(ROOT/'inputs')
visit=json.loads(plan_path.read_text())
assert visit['physical_gpus']==[0] and not visit.get('admission_hold', False)
assert visit['requests'] and all(x['role'] in ('admission','warmup','measure') for x in visit['requests'])
folder=ROOT/'runs'/run
folder.mkdir(parents=True,exist_ok=False)
(folder/'visit.json').write_bytes(plan_path.read_bytes())
container='bbuf-diffusion-6000-20261113-'+arm
python='/opt/sglang/bin/python' if arm=='sgl' else '/task/envs/omni/bin/python'
manifest=json.loads((ROOT/'manifest.json').read_text())
default_repo='/task/repos/sglang-main' if arm=='sgl' else '/task/repos/vllm-omni'
repo=visit.get('repository',default_repo)
allowed={'/task/repos/sglang-main':manifest['sglang_sha'], '/task/repos/vllm-omni':manifest['omni_sha'], '/task/candidates/qwen21-bcg-prefix':'4b45ef7fd5b3106e577634561e7f6d5911632692', '/task/candidates/qwen21-bcg-prefix-release':'15b28203ba81c96fac1b3692d7bc67bac01d8ff5'}
expected_head=allowed[repo]
assert (arm=='omni') == (repo=='/task/repos/vllm-omni')
assert visit.get('head_sha',expected_head)==expected_head
if repo.startswith('/task/candidates/'):
    assert visit['server_env']['sgl']['PYTHONPATH'].split(':')[-1]==repo+'/python'
state=dict(arm=arm,run=run,purpose=visit['purpose'],benchmark=any(x['role']=='measure' for x in visit['requests']),
    status='preflight',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=[])
state['profiler']=any(x.get('profile') for x in visit['requests'])
assert not (state['benchmark'] and state['profiler']), 'Profile and benchmark require separate visits'


def save(**kw):
    state.update(kw);state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (folder/'status.json').write_text(json.dumps(state,indent=2)+'\n')


def output(cmd):
    return subprocess.check_output(cmd,text=True,timeout=30).strip()


known_gpu_processes={}


def process_identity(pid):
    path=Path(f'/proc/{pid}/stat')
    try:
        fields=path.read_text().rsplit(')',1)[1].split()
        return dict(start_ticks=fields[19],cgroup=Path(f'/proc/{pid}/cgroup').read_text())
    except FileNotFoundError:
        return None


def observe(idle=False, cleanup=False):
    gpu=output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader,nounits'])
    assert [x.split(', ')[1] for x in gpu.splitlines()]==[g['uuid'] for g in manifest['gpu_details']]
    pids={int(x) for x in output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).split()}
    owned={int(x) for x in output(['docker','top',container,'-eo','pid']).splitlines()[1:]}
    for pid in pids & owned:
        identity=process_identity(pid)
        if identity is not None:
            assert pid not in known_gpu_processes or known_gpu_processes[pid]==identity, 'Owned GPU PID reused'
            known_gpu_processes[pid]=identity
    foreign=pids-owned
    if cleanup:
        foreign={pid for pid in foreign if pid not in known_gpu_processes or process_identity(pid) not in (None,known_gpu_processes[pid])}
    with (folder/'gpu-samples.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),gpu=gpu,gpu_pids=sorted(pids),owned_pids=sorted(owned),known_gpu_processes=known_gpu_processes,cleanup=cleanup))+'\n')
    assert not foreign,f'foreign GPU processes {foreign}'
    if idle:assert not pids,f'GPUs occupied {pids}'
    return pids


server=client=None
try:
    observe(idle=True)
    assert output(['docker','exec',container,'git','-C',repo,'rev-parse','HEAD'])==expected_head
    assert not output(['docker','exec',container,'git','-C',repo,'status','--porcelain'])
    check=json.loads((ROOT/'evidence/preflight/qwen-image-2.1-cache-fresh-verification.json').read_text())
    assert visit['model'] == 'Qwen/Qwen-Image-2.1'
    assert check['complete'] and check['revision'] == visit['weights_revision']
    assert len(check['files']) == 24 and all(x['verified'] for x in check['files'])
    assert json.loads((ROOT/'evidence/preflight/qwen21-cpu-contract-comparison.json').read_text())['passed']
    if state['benchmark']:
        gate_path=(ROOT/visit['quality_gate_metrics']).resolve()
        assert gate_path.is_relative_to(ROOT/'evidence/quality')
        assert json.loads(gate_path.read_text())['numeric_passed']
        assert all(x.get('expected_sha256',{}).get(arm) for x in visit['requests']), 'measurement requires this configuration admitted output SHA'
    helper='/task/scripts/h3-native-visits/diffusion_skill_env.py'
    sgl_repo=repo if arm=='sgl' else '/task/repos/sglang-main'
    helper_command=['docker','exec','-e','PYTHONPATH='+sgl_repo+'/python','bbuf-diffusion-6000-20261113-sgl','/opt/sglang/bin/python',helper]
    assert output([*helper_command,'print-root'])==sgl_repo
    output([*helper_command,'check-write-access'])
    assert output([*helper_command,'print-idle-gpus','--count','1'])=='0'
    save(repository=repo,head_sha=expected_head,main_baseline_sha=manifest['sglang_sha'])
    (folder/'pip-freeze.txt').write_text(output(['docker','exec',container,'uv','pip','freeze','--python',python])+'\n')
    (folder/'host-meminfo.txt').write_text(Path('/proc/meminfo').read_text())
    snapshot=folder/'reproduction';snapshot.mkdir();hashes={}
    files=[ROOT/'manifest.json',ROOT/'inputs/qwen21-native-workloads.json',plan_path,*sorted((ROOT/'scripts/qwen21-codec0-visits').glob('*.py'))]
    for p in files:
        data=p.read_bytes();(snapshot/p.name).write_bytes(data);hashes[str(p.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
    launch=['docker','exec',container,python,'/task/scripts/qwen21-codec0-visits/serve.py','start',arm,run]
    state['commands'].append(launch);save(status='starting',input_hashes=hashes)
    start=time.monotonic()
    with (folder/'server.log').open('xb') as log:server=subprocess.Popen(launch,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    endpoint='http://127.0.0.1:'+('32150/liveness' if arm=='sgl' else '32151/health')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while True:
        assert server.poll() is None,f'server exited {server.returncode}'
        observe();assert time.monotonic()-start<5400,'startup deadline'
        try:
            with opener.open(endpoint,timeout=2) as response:ready=response.status==200
        except (OSError,urllib.error.URLError):ready=False
        log_text=(folder/'server.log').read_text(errors='replace')
        assert not any(x in log_text for x in ('Falling back to diffusers backend','Using diffusers backend','Loaded diffusers pipeline'))
        if arm=='sgl':ready=ready and ('Warmup requests: 100%' in log_text or bool(re.search(r'Warmup requests: ([1-9]\d*)/\1\s',log_text)))
        if ready:break
        time.sleep(2)
    save(status='requests',launch_to_ready_seconds=time.monotonic()-start)
    command=['docker','exec',container,python,'/task/scripts/qwen21-codec0-visits/submit.py',arm,run]
    state['commands'].append(command)
    with (folder/'requests.log').open('xb') as log:client=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    start=time.monotonic()
    while client.poll() is None:
        observe();assert server.poll() is None,'server exited during requests'
        assert time.monotonic()-start<7200*len(visit['requests']),'visit deadline'
        time.sleep(2)
    assert client.returncode==0,f'client exited {client.returncode}'
    rows=json.loads((folder/'outputs/results.json').read_text());assert len(rows)==len(visit['requests'])
    if state['profiler']:
        traces=list((folder/'profiles').glob('*.trace.json.gz'))
        assert traces and all(p.stat().st_size>0 for p in traces), 'Missing native profiler trace'
        save(profile_traces=[str(p.relative_to(ROOT)) for p in traces])
    if visit.get('require_own_repeat'):
        for prompt in {x['prompt_id'] for x in rows}:
            assert len({x['sha256'] for x in rows if x['prompt_id']==prompt})==1,f'own repeat changed: {prompt}'
    save(status='raw_complete',completed_outputs=len(rows),reference_checks=[x['matches_reference'] for x in rows])
except BaseException as exc:
    save(status='failed_preserve_no_retry',error=repr(exc));raise
finally:
    if server is not None and server.poll() is None and (folder/'server-identity.json').exists():
        stop=['docker','exec',container,python,'/task/scripts/qwen21-codec0-visits/serve.py']
        subprocess.run([*stop,'term',arm,run],check=True,timeout=30)
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            subprocess.run([*stop,'kill',arm,run],check=True,timeout=30);server.wait(timeout=30)
    if client is not None and client.poll() is None:
        try:client.wait(timeout=30)
        except subprocess.TimeoutExpired:save(client_cleanup='still alive; inspect owned container')
    try:
        deadline=time.monotonic()+60
        while True:
            time.sleep(2)
            if not observe(cleanup=True):
                save(gpu_cleanup='idle_verified');break
            assert time.monotonic()<deadline, 'Known GPU workers still reported after60s; inspect without restarting'
    except Exception as exc:save(gpu_cleanup='not_idle_requires_inspection',cleanup_error=repr(exc))
