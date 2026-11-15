"""Finish frozen single-GPU241-frame native admission without timing claims."""
import datetime
import json
import subprocess
from pathlib import Path

root=Path('/data/xyzhang/diffusion-6000-20261113')
status_path=root/'evidence/preflight/nomas-matrix17-controller.json'
assert not status_path.exists()
state=dict(status='preflight',benchmark=False,completed=[],physical_gpus=[0])


def save(**updates):
    state.update(updates,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status_path.write_text(json.dumps(state,indent=2)+'\n')


def visit(arm,name,script,plan):
    command=['python3',str(root/'scripts'/script/'run.py'),arm,name,plan]
    save(status='running',active=name,command=command)
    subprocess.run(command,check=True)
    folder=root/'runs'/name
    status=json.loads((folder/'status.json').read_text())
    assert status['status']=='raw_complete' and status['gpu_cleanup']=='idle_verified',status
    rows=json.loads((folder/'outputs/results.json').read_text())
    assert len(rows)==7 and rows[0]['sha256']==rows[-1]['sha256']
    state['completed'].append(name)
    save()
    return rows


try:
    assert json.loads((root/'evidence/preflight/nomas-scope16-controller.json').read_text())['status']=='complete'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    main=visit('sgl','nomas-241-sgl-native17','nomas-native-visits02','inputs/nomas-241-sgl-native17.json')
    visit('omni','nomas-241-omni-native17','nomas-native-visits02','inputs/nomas-241-omni-native17.json')
    plan=json.loads((root/'inputs/nomas-241-PR-template17.json').read_text())
    for item in plan['requests']:
        item['expected_sha256']={'sgl':next(a['sha256'] for a in main if a['workload_id']==item['id'])}
    name='nomas-241-sgl-mean-quality17'
    target=root/'inputs'/f'{name}.json'
    assert not target.exists()
    target.write_text(json.dumps(plan,indent=2)+'\n')
    visit('sgl',name,'nomas-cleanup-long-visits13','inputs/'+name+'.json')
    command=['docker','exec','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONPATH=/task/envs/quality-tools',
             'bbuf-diffusion-6000-20261113-sgl','/opt/sglang/bin/python','/task/scripts/audit_nomas_native17.py']
    save(active='native241_cross_quality_CPU',command=command)
    subprocess.run(command,check=True)
    state['completed'].append('native241_cross_quality_CPU')
    save(status='complete',active=None)
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry',error=repr(exc))
    raise
