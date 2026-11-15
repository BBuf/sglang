"""Resume only the unrun high candidate and121 quality after a plan-write error."""
import datetime
import json
import subprocess
from pathlib import Path

root=Path('/data/xyzhang/diffusion-6000-20261113')
path=root/'evidence/preflight/nomas-scope16-controller.json';assert not path.exists()
state=dict(status='preflight',completed=[],benchmark=False)


def save(**values):
    state.update(values,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state,indent=2)+'\n')


try:
    prior=json.loads((root/'evidence/preflight/nomas-scope15-controller.json').read_text())
    assert prior['status']=='failed_preserve_no_automatic_retry' and 'PermissionError' in prior['error']
    assert prior['completed']==['fallback_main','fallback_PR','nomas-49-high-main-scope15']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    main_folder=root/'runs/nomas-49-high-main-scope15'
    assert json.loads((main_folder/'status.json').read_text())['gpu_cleanup']=='idle_verified'
    main=json.loads((main_folder/'outputs/results.json').read_text())
    assert len(main)==3 and main[0]['sha256']==main[-1]['sha256']
    plan=json.loads((root/'inputs/nomas-49-high-PR-scope15.json').read_text())
    name='nomas-49-high-PR-scope16'
    plan['server_env']['sgl']['NOMAS_AUDIT_DIR']='/task/runs/'+name+'/runtime-audit'
    for item in plan['requests']:
        item['expected_sha256']={'sgl':next(x['sha256'] for x in main if x['workload_id']==item['id'])}
    plan['reference_hash_source']='runs/nomas-49-high-main-scope15/outputs/results.json'
    target=root/'inputs'/f'{name}.json';assert not target.exists()
    target.write_text(json.dumps(plan,indent=2)+'\n')
    command=['python3',str(root/'scripts/nomas-scope-visits15/run.py'),'sgl',name,'inputs/'+name+'.json']
    save(status='running',active=name,command=command)
    subprocess.run(command,check=True)
    status=json.loads((root/'runs'/name/'status.json').read_text())
    assert status['status']=='raw_complete' and status['gpu_cleanup']=='idle_verified',status
    state['completed'].append(name)
    command=['docker','exec','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONPATH=/task/envs/quality-tools',
             'bbuf-diffusion-6000-20261113-sgl','/opt/sglang/bin/python','/task/scripts/audit_nomas_native09.py']
    save(active='native121_cross_quality_CPU',command=command)
    subprocess.run(command,check=True)
    state['completed'].append('native121_cross_quality_CPU')
    save(status='complete',active=None)
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry',error=repr(exc))
    raise
