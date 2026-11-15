"""Serialize cleaned-candidate checks and longer native admissions on GPU 0."""
import datetime
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-6000-20261113')
path = root / 'evidence/preflight/nomas-validation11-controller.json'
assert not path.exists()
state = dict(status='waiting_for_noise10', completed=[], benchmark=False,
             cross_framework_performance_admitted=False)


def save(**values):
    state.update(values, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state, indent=2)+'\n')


try:
    deadline = time.monotonic() + 1200
    while True:
        prior = json.loads((root/'evidence/preflight/nomas-noise10-controller.json').read_text())
        assert not prior['status'].startswith('failed'), prior
        if prior['status'] == 'complete':
            break
        assert time.monotonic() < deadline
        save()
        time.sleep(10)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    repo = '/task/candidates/nomas-vae-mean-cleanup'
    base = ['docker','exec','-e','CUDA_VISIBLE_DEVICES=0','-e','FLASHINFER_DISABLE_VERSION_CHECK=1',
            '-e','PYTHONPATH='+repo+'/python','bbuf-diffusion-6000-20261113-sgl','/opt/sglang/bin/python']
    helper = '/task/scripts/h3-native-visits/diffusion_skill_env.py'
    assert subprocess.check_output(base+[helper,'print-root'],text=True).strip() == repo
    subprocess.run(base+[helper,'check-write-access'],check=True)
    assert subprocess.check_output(base+[helper,'print-idle-gpus','--count','1'],text=True).strip() == '0'
    command = base+['-m','pytest',repo+'/test/registered/kernels/ops/diffusion/test_norm.py','-k','mean_square','-q']
    save(status='running', active_run='cleaned_kernel_tests', command=command)
    with (root/'evidence/preflight/nomas-mean11-kernel-tests.log').open('xb') as log:
        test = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    result = dict(command=command,returncode=test.returncode,source_sha='37999f7ab31809bd8a9f3cabee3fa50c54f179d6',physical_gpus=[0])
    (root/'evidence/preflight/nomas-mean11-kernel-tests.json').write_text(json.dumps(result,indent=2)+'\n')
    assert test.returncode == 0
    state['completed'].append('cleaned_kernel_tests')
    for arm, name, harness in [
        ('sgl','nomas-49-sgl-mean-quality11','nomas-cleanup-visits11'),
        ('sgl','nomas-49-sgl-mean-profile11','nomas-cleanup-profile11'),
        ('sgl','nomas-121-sgl-native09','nomas-native-visits02'),
        ('omni','nomas-121-omni-native09','nomas-native-visits02'),
    ]:
        command = ['python3',str(root/'scripts'/harness/'run.py'),arm,name,'inputs/'+name+'.json']
        save(active_run=name,command=command)
        subprocess.run(command,check=True)
        status = json.loads((root/'runs'/name/'status.json').read_text())
        assert status['status']=='raw_complete' and status['gpu_cleanup']=='idle_verified', status
        state['completed'].append(name)
        save()
    save(status='complete',active_run=None)
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry',error=repr(exc))
    raise
