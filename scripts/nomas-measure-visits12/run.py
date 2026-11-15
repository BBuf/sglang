"""Unprofiled native main noise visits; cross-framework comparisons stay excluded."""
import datetime
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path('/data/xyzhang/diffusion-6000-20261113')
arm,run,plan=sys.argv[1:]
assert arm=='sgl' and Path(run).name==run
plan_path=(ROOT/plan).resolve()
assert plan_path.is_relative_to(ROOT/'inputs')
visit=json.loads(plan_path.read_text())
assert visit['physical_gpus'] in ([0],[0,1]) and 0<visit['poll_seconds']<=1
assert visit['requests'] and all(x['role'] in ('admission','warmup','measure') for x in visit['requests'])
folder=ROOT/'runs'/run
folder.mkdir(parents=True,exist_ok=False)
(folder/'visit.json').write_bytes(plan_path.read_bytes())
container='bbuf-diffusion-6000-20261113-'+arm
python='/opt/sglang/bin/python' if arm=='sgl' else '/task/envs/omni/bin/python'
repo=visit['source_repo']
assert arm=='sgl' and repo in ('/task/repos/sglang-main','/task/candidates/nomas-vae-mean','/task/candidates/nomas-vae-mean-cleanup')
manifest=json.loads((ROOT/'manifest.json').read_text())
state=dict(arm=arm,run=run,purpose=visit['purpose'],benchmark=any(x['role']=='measure' for x in visit['requests']),
    status='preflight',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=[])


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


assert state['benchmark'] and visit['benchmark']
quality=json.loads((ROOT/'evidence/quality/nomas-49-native02/summary.json').read_text())
own_repeat=next(x for x in quality['pairs'] if x['name']=='sgl-own-repeat')
assert own_repeat['numeric_passed'] and own_repeat['artifact_exact']
assert visit['comparison_scope'] in ('within_sglang_main_candidate_final','within_sglang_cleanup','within_sglang_stage_sync_diagnostic') and not visit['cross_framework_performance_admitted']
state.update(comparison_scope=visit['comparison_scope'],cross_framework_performance_admitted=False,
    cross_framework_quality_passed=quality['numeric_passed'],
    quality_evidence='evidence/quality/nomas-49-native02/summary.json')
assert len(visit['requests'])==8 and not any(x['profile'] for x in visit['requests'])
assert [x['role'] for x in visit['requests']]==['warmup']*2+['measure']*6
assert not any('NOMAS_AUDIT' in k for k in visit['server_env'][arm])
assert visit['server_env'][arm]['PYTHONPATH']==repo+'/python'
admission=ROOT/visit['quality_admission']
verified=json.loads((admission/'status.json').read_text())
assert verified['status']=='raw_complete' and verified['gpu_cleanup']=='idle_verified'
assert all(x['matches_reference'] for x in json.loads((admission/'outputs/results.json').read_text()))
if repo != '/task/repos/sglang-main':
    assert json.loads((admission/'visit.json').read_text())['source_sha']==visit['source_sha']
reference=json.loads((ROOT/'runs/nomas-49-sgl-native02/outputs/results.json').read_text())
for item in visit['requests']:
    row=next(x for x in reference if x['workload_id']==item['id'])
    assert item['expected_sha256'][arm]==row['sha256']
server=client=None
try:
    observe(idle=True)
    assert output(['docker','exec',container,'git','-C',repo,'rev-parse','HEAD'])==visit['source_sha']
    assert not output(['docker','exec',container,'git','-C',repo,'status','--porcelain'])
    check=json.loads((ROOT/'evidence/preflight/nomas-common-weight-identity01.json').read_text())
    assert len(check['files_freshly_verified'])==38 and len(check['text_encoder_tensor_identity'])==338
    adaptation=json.loads((ROOT/'evidence/preflight/nomas-key-only-materialization02.json').read_text())
    assert len(adaptation['files'])==4 and all(x['numeric_payload_byte_exact'] for x in adaptation['files'])
    assert not adaptation['numeric_dtype_or_layout_changed']
    mask=json.loads((ROOT/'evidence/preflight/nomas-action-mask-cpu01-comparison.json').read_text())
    assert mask['all21_equal']
    helper='/task/scripts/h3-native-visits/diffusion_skill_env.py'
    assert output(['docker','exec','-e','PYTHONPATH='+repo+'/python',container,python,helper,'print-root'])==repo
    output(['docker','exec','-e','PYTHONPATH='+repo+'/python',container,python,helper,'check-write-access'])
    assert output(['docker','exec','-e','PYTHONPATH='+repo+'/python',container,python,helper,'print-idle-gpus','--count',str(len(visit['physical_gpus']))])==','.join(map(str,visit['physical_gpus']))
    (folder/'pip-freeze.txt').write_text(output(['docker','exec',container,'uv','pip','freeze','--python',python])+'\n')
    (folder/'host-meminfo.txt').write_text(Path('/proc/meminfo').read_text())
    snapshot=folder/'reproduction';snapshot.mkdir();hashes={}
    files=[ROOT/'manifest.json',ROOT/'inputs/nomas-native-workloads01.json',plan_path,*sorted((ROOT/'scripts/nomas-measure-visits12').glob('*.py'))]
    for p in files:
        data=p.read_bytes();(snapshot/p.name).write_bytes(data);hashes[str(p.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
    launch=['docker','exec',container,python,'/task/scripts/nomas-measure-visits12/serve.py','start',arm,run]
    state['commands'].append(launch);save(status='starting',input_hashes=hashes,source_repo=repo,source_sha=visit['source_sha'])
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
        if ready:break
        time.sleep(2)
    save(status='requests',launch_to_ready_seconds=time.monotonic()-start)
    command=['docker','exec',container,python,'/task/scripts/nomas-measure-visits12/submit.py',arm,run]
    state['commands'].append(command)
    with (folder/'requests.log').open('xb') as log:client=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    start=time.monotonic()
    while client.poll() is None:
        observe();assert server.poll() is None,'server exited during requests'
        assert time.monotonic()-start<7200*len(visit['requests']),'visit deadline'
        time.sleep(2)
    assert client.returncode==0,f'client exited {client.returncode}'
    rows=json.loads((folder/'outputs/results.json').read_text());assert len(rows)==len(visit['requests'])
    if visit.get('require_own_repeat'):
        for prompt in {x['prompt_id'] for x in rows}:
            assert len({x['sha256'] for x in rows if x['prompt_id']==prompt})==1,f'own repeat changed: {prompt}'
    assert not list((folder/'profiles').glob('*'))
    save(status='raw_complete',completed_outputs=len(rows),reference_checks=[x['matches_reference'] for x in rows])
except BaseException as exc:
    save(status='failed_preserve_no_retry',error=repr(exc));raise
finally:
    if server is not None and server.poll() is None and (folder/'server-identity.json').exists():
        stop=['docker','exec',container,python,'/task/scripts/nomas-measure-visits12/serve.py']
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
