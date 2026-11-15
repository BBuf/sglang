"""Run native two-GPU admission after the single-GPU241 matrix job finishes."""
import datetime
import json
import subprocess
import time
from pathlib import Path

root=Path('/data/xyzhang/diffusion-6000-20261113')
path=root/'evidence/preflight/nomas-sp2_18-controller.json'
assert not path.exists()
state=dict(status='waiting_matrix17',benchmark=False,completed=[],physical_gpus=[0,1])


def save(**updates):
    state.update(updates,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state,indent=2)+'\n')


try:
    save()
    deadline=time.monotonic()+3*3600
    while True:
        previous=json.loads((root/'evidence/preflight/nomas-matrix17-controller.json').read_text())
        if previous['status']=='complete':break
        assert not previous['status'].startswith('failed'),previous
        assert time.monotonic()<deadline,'prior job deadline; inspect without retry'
        time.sleep(15)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    main=None
    for arm in ['sgl','omni','PR']:
        native='omni' if arm=='omni' else 'sgl'
        name=f'nomas-sp2-{arm}-native18'
        if arm=='PR':
            plan=json.loads((root/'inputs'/f'{name}-template.json').read_text())
            for item in plan['requests']:
                item['expected_sha256']={'sgl':next(x['sha256'] for x in main if x['workload_id']==item['id'])}
            target=root/'inputs'/f'{name}.json';assert not target.exists()
            target.write_text(json.dumps(plan,indent=2)+'\n')
        command=['python3',str(root/'scripts/nomas-sp2-visits18/run.py'),native,name,'inputs/'+name+'.json']
        save(status='running',active=name,command=command)
        subprocess.run(command,check=True)
        folder=root/'runs'/name
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='raw_complete' and status['gpu_cleanup']=='idle_verified',status
        rows=json.loads((folder/'outputs/results.json').read_text())
        assert len(rows)==6 and rows[0]['sha256']==rows[2]['sha256'] and rows[3]['sha256']==rows[5]['sha256']
        if arm=='sgl':main=rows
        state['completed'].append(name);save()
    command=['docker','exec','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONPATH=/task/envs/quality-tools',
             'bbuf-diffusion-6000-20261113-sgl','/opt/sglang/bin/python','/task/scripts/audit_nomas_sp2_18.py']
    save(active='SP2_cross_quality_CPU',command=command)
    subprocess.run(command,check=True)
    state['completed'].append('SP2_cross_quality_CPU')
    save(status='complete',active=None)
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry',error=repr(exc))
    raise
