"""Explicit serial same-GPU blocks. Plans pin source, order, quality and repetitions."""
import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path
ROOT=Path('/data/xyzhang/diffusion-rtx6000-20261026')
BASE='af21e2b989e482cfdce39689126562f2209d53ac'
OMNI='817d4c17feb702289a2d2b2cbacfd594c262b7a7'
EMPTY=hashlib.sha256(b'').hexdigest()
def command(args):return subprocess.check_output(args,text=True).strip()
def idle():return not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
def run_block(plan,index,config):
 spec=plan['configurations'][config];arm=spec['arm'];repo=spec['repo'];sha=spec.get('head',BASE if arm=='sgl' else OMNI);diffsha=spec.get('diff_sha256',EMPTY)
 assert plan['SGLANG_DIFFUSION_SYNC_STAGE_PROFILING']==0 and not plan.get('diagnostic_timing')
 name=f'{plan["id"]}-{index:02d}-{config}';folder=ROOT/'block-runs'/name;folder.mkdir(exist_ok=False)
 container='bbuf-diffusion-rtx6000-20261026-'+arm;port=31050 if arm=='sgl' else 31051
 assert idle(),'Unowned GPU work present'
 assert command(['docker','exec',container,'git','-C',repo,'rev-parse','HEAD'])==sha
 patch=subprocess.check_output(['docker','exec',container,'git','-C',repo,'diff','HEAD','--binary','--full-index']);assert hashlib.sha256(patch).hexdigest()==diffsha
 (folder/'source.patch').write_bytes(patch)
 preflight_repo=repo if arm=='sgl' else '/task/repos/sglang'
 with (folder/'preflight.log').open('xb') as stream:
  subprocess.run(['docker','exec','-e','PYTHONPATH='+preflight_repo+'/python','-e','FLASHINFER_DISABLE_VERSION_CHECK=1','bbuf-diffusion-rtx6000-20261026-sgl','python','/task/preflight_diffusion_current_head1.py',preflight_repo],stdout=stream,stderr=subprocess.STDOUT,check=True)
 script=spec['launcher'];(folder/'launch.sh').write_bytes((ROOT/script).read_bytes())
 launch=['docker','exec','-e',f'QWEN_CANDIDATE_SHA={sha}','-e',f'SGL_REPO={repo}','-e',f'QWEN_CANDIDATE_DIFF_SHA={diffsha}',container,'sh','-c',f'echo $$ > /task/block-runs/{name}/server.pid; exec sh /task/{script}']
 meta=dict(plan=plan,index=index,config=config,source_sha=sha,source_diff_sha256=diffsha,launch_command=launch,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),commands=[])
 def save(): (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 def execute(args,label):
  meta['commands'].append(args);save()
  with (folder/(label+'.log')).open('xb') as stream:subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=7200)
 save();start=time.monotonic();log=folder/'server.log'
 with log.open('xb') as stream:server=subprocess.Popen(launch,stdout=stream,stderr=subprocess.STDOUT)
 def native_gate():
  if arm=='sgl':
   contents=log.read_text(errors='replace')
   assert not any(marker in contents for marker in ['Falling back to diffusers backend','Using diffusers backend','Loaded diffusers pipeline']),'Native backend gate failed'
 try:
  while True:
   native_gate()
   if server.poll() is not None:raise RuntimeError(f'Server exit {server.returncode}: {log}')
   if time.monotonic()-start>1800:raise TimeoutError('readiness')
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}'+('/liveness' if arm=='sgl' else '/health'),timeout=2) as response:ready=response.status==200
   except Exception:ready=False
   if arm=='sgl':ready=ready and b'Warmup requests: 100%' in log.read_bytes()
   if ready:break
   time.sleep(2)
  meta['launch_to_ready_seconds']=time.monotonic()-start
  q='import json,pynvml as n;n.nvmlInit();print(json.dumps([[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(n.nvmlDeviceGetHandleByIndex(i))] for i in (0,1)]))'
  pids=json.loads(command(['docker','exec',container,'python','-c',q]));assert pids[0] and not pids[1];meta['gpu_container_pids']=pids
  for task in plan.get('task_orders',{}).get(str(index),spec.get('tasks',plan['tasks'])):
   contract=ROOT/'inputs'/plan.get('contracts',{}).get(task,f'qwen21-base-{task}-workloads.json');assert hashlib.sha256(contract.read_bytes()).hexdigest()==plan['contract_sha256'][task]
   label=f'base-{task}-{name}-{plan["quality"]}';qualified=f'{arm}-qwen21-{label}'
   execute(['docker','exec',container,'python','/task/submit_qwen21_candidate_qualification.py',arm,'--label',label,'--quality',plan['quality'],'--contract',f'/task/inputs/{contract.name}','--repo',repo,'--expected-sha',sha,'--expected-diff-sha256',diffsha],task+'-qualification')
   refs=plan['quality_references'][config][task]
   cmd=['docker','exec',container,'python','/task/compare_qwen21_qualification_v2.py',qualified]
   for ref in refs:cmd+=['--reference',ref]
   if refs:execute(cmd,task+'-quality')
   else:assert not plan.get('benchmark',True),'No unqualified primary measurement'
   if plan.get('require_sgl_pixel_exact') and arm=='sgl':
    sgl_refs=[ref for ref in refs if ref.startswith('sgl-')]
    assert sgl_refs
    exact=['docker','exec',container,'python','/task/compare_qwen21_qualification_v2.py',qualified,'--require-pixel-identical','--output',f'/task/outputs/qualification/{qualified}-sgl-exact-quality.json']
    for ref in sgl_refs:exact+=['--reference',ref]
    execute(exact,task+'-sgl-exact-quality')
   if not plan.get('benchmark',True):continue
   runid=f'{name}-{task}'
   execute(['docker','exec',container,'python','/task/bench_qwen21_serving.py',arm,'--run-id',runid,'--repo',repo,'--allowed-gpu-pids',*map(str,pids[0]),'--quality',plan['quality'],'--configuration-label',config,'--warmups',str(plan['warmups']),'--repeats',str(plan['repeats']),'--expected-sha',sha,'--expected-diff-sha256',diffsha,'--contract',f'/task/inputs/{contract.name}']+(['--diagnostic-profiler'] if plan.get('diagnostic_timing') else []),task+'-benchmark')
   actual=json.loads((ROOT/'outputs/benchmarks'/runid/'results.json').read_text());ref=json.loads((ROOT/'outputs/qualification'/qualified/'results.json').read_text())[0]['sha256']
   assert all(r['sha256']==ref for r in actual),'Measured outputs differ from same-process qualified output'
  native_gate()
  meta['status']='complete'
 except Exception as e:
  meta.update(status='failed',error=repr(e));raise
 finally:
  save();pid=folder/'server.pid'
  if server.poll() is None and pid.exists():subprocess.run(['docker','exec',container,'kill','-TERM',pid.read_text().strip()],check=True)
  server.wait(timeout=60)
  for _ in range(30):
   if idle():break
   time.sleep(1)
  assert idle(),'GPU work remains; next block prohibited'
 return name
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('plan',type=Path);args=p.parse_args();plan=json.loads(args.plan.read_text());status=ROOT/(plan['id']+'-status.json');assert not status.exists();state=dict(status='running',completed=[])
 for path,expected in plan['sha256'].items():
  actual=ROOT/(Path(path).name if path.startswith('scripts/') else path)
  assert hashlib.sha256(actual.read_bytes()).hexdigest()==expected,path
 for path in plan['required_admissions']:
  assert json.loads((ROOT/path).read_text())['passed'],path
 receipt=json.loads((ROOT/'evidence/checkpoints/joy-input-lifetime-repair1-diagnostics1-local-verification.json').read_text())
 assert receipt['all_member_hashes_verified'] and receipt['all_members_extracted']
 assert plan['quality']=='high' and plan['benchmark'] and plan['SGLANG_DIFFUSION_SYNC_STAGE_PROFILING']==0
 try:
  for i,config in enumerate(plan['order']):
   state.update(active_index=i,active_configuration=config);status.write_text(json.dumps(state,indent=2)+'\n')
   state['completed'].append(run_block(plan,i,config))
  state.update(status='raw_complete_audit_pending',active_index=None,active_configuration=None)
 except Exception as e:state.update(status='failed',error=repr(e));raise
 finally:status.write_text(json.dumps(state,indent=2)+'\n')
