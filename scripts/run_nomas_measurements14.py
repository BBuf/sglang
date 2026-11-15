"""After isolated validation, test longer output identity and final/cleanup ABBA."""
import datetime
import json
import subprocess
from pathlib import Path

root = Path('/data/xyzhang/diffusion-6000-20261113')
path = root/'evidence/preflight/nomas-measurements14-controller.json'
assert not path.exists()
state = dict(status='preflight',completed=[],cross_framework_performance_admitted=False)


def save(**values):
    state.update(values,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state,indent=2)+'\n')


def visit(name, harness, arm='sgl'):
    command=['python3',str(root/'scripts'/harness/'run.py'),arm,name,'inputs/'+name+'.json']
    save(status='running',active_run=name,command=command)
    subprocess.run(command,check=True)
    status=json.loads((root/'runs'/name/'status.json').read_text())
    assert status['status']=='raw_complete' and status['gpu_cleanup']=='idle_verified',status
    state['completed'].append(name)
    save()


try:
    reconciliation=json.loads((root/'evidence/preflight/nomas-profile11-reconciliation.json').read_text())
    assert reconciliation['status']=='profile_complete_reconciled' and reconciliation['outputs_exact']
    for controller in ['nomas-validation11-controller.json','nomas-measurements12-controller.json']:
        assert json.loads((root/'evidence/preflight'/controller).read_text())['status']=='failed_preserve_no_automatic_retry'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for arm in ['sgl','omni']:
        name='nomas-121-'+arm+'-native09'
        assert not (root/'runs'/name).exists()
        visit(name,'nomas-native-visits02',arm)
    # Derive only output-reference hashes from the completed real main run.
    name='nomas-121-sgl-mean-quality13'
    plan=json.loads((root/'inputs/nomas-121-sgl-native09.json').read_text())
    main=json.loads((root/'runs/nomas-121-sgl-native09/outputs/results.json').read_text())
    assert len(main)==7 and main[0]['sha256']==main[-1]['sha256']
    plan.update(source_repo='/task/candidates/nomas-vae-mean-cleanup',
                source_sha='37999f7ab31809bd8a9f3cabee3fa50c54f179d6',
                purpose='Cleaned candidate121frame exact native-main output and2166actualfusion checks; no performance claim',
                expected_mean_square_calls=2166)
    plan['server_env']['sgl'].update(PYTHONPATH='/task/scripts/nomas-vae-mean-observer07:/task/candidates/nomas-vae-mean-cleanup/python',
                                   NOMAS_AUDIT_DIR='/task/runs/'+name+'/runtime-audit')
    for item in plan['requests']:
        row=next(x for x in main if x['workload_id']==item['id'])
        item['expected_sha256']={'sgl':row['sha256']}
    target=root/'inputs'/f'{name}.json'
    assert not target.exists()
    target.write_text(json.dumps(plan,indent=2)+'\n')
    visit(name,'nomas-cleanup-long-visits13')
    noise=json.loads((root/'evidence/performance/nomas-main-noise10.json').read_text())
    assert noise['poll_seconds']==.02 and noise['conservative_noise_bound_pct']<1.5
    queue=json.loads((root/'inputs/nomas-measurements12-queue.json').read_text())
    for name in queue['visits']:
        visit(name,'nomas-measure-visits12')
    save(status='complete',active_run=None)
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry',error=repr(exc))
    raise
