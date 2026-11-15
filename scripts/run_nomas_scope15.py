"""Run compatibility and CPU cross-quality only after quiet measurements end."""
import datetime
import json
import subprocess
import time
from pathlib import Path

root=Path('/data/xyzhang/diffusion-6000-20261113')
path=root/'evidence/preflight/nomas-scope15-controller.json';assert not path.exists()
state=dict(status='waiting_for_measurements14',completed=[],benchmark=False)


def save(**values):
    state.update(values,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state,indent=2)+'\n')


try:
    deadline=time.monotonic()+7200
    while True:
        previous=json.loads((root/'evidence/preflight/nomas-measurements14-controller.json').read_text())
        assert not previous['status'].startswith('failed'),previous
        if previous['status']=='complete':break
        assert time.monotonic()<deadline
        save();time.sleep(10)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for arm,repo in [('main','/task/repos/sglang-main'),('PR','/task/candidates/nomas-vae-mean-cleanup')]:
        command=['docker','exec','-e','CUDA_VISIBLE_DEVICES=0','-e','FLASHINFER_DISABLE_VERSION_CHECK=1',
                 '-e','PYTHONPATH='+repo+'/python','bbuf-diffusion-6000-20261113-sgl','/opt/sglang/bin/python',
                 '/task/scripts/check_nomas_fallback15.py','/task/evidence/preflight/nomas-fallback15-'+arm+'.json']
        save(status='running',active='fallback_'+arm,command=command)
        subprocess.run(command,check=True)
        state['completed'].append('fallback_'+arm)
    for arm in ['main','PR']:
        name='nomas-49-high-'+arm+'-scope15'
        if arm=='PR':
            p=root/'inputs'/f'{name}.json';plan=json.loads(p.read_text())
            main=json.loads((root/'runs/nomas-49-high-main-scope15/outputs/results.json').read_text())
            assert len(main)==3 and main[0]['sha256']==main[-1]['sha256']
            for item in plan['requests']:
                item['expected_sha256']={'sgl':next(x['sha256'] for x in main if x['workload_id']==item['id'])}
            plan['reference_hash_source']='runs/nomas-49-high-main-scope15/outputs/results.json'
            p.write_text(json.dumps(plan,indent=2)+'\n')
        command=['python3',str(root/'scripts/nomas-scope-visits15/run.py'),'sgl',name,'inputs/'+name+'.json']
        save(active=name,command=command);subprocess.run(command,check=True)
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
