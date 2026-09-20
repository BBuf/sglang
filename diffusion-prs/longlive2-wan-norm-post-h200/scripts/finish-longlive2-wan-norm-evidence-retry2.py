"""Validate the corrected fallback, repeat native A/B, then audit final evidence."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
root=Path('/campaign');repo=root/'candidate-longlive2-wan-norm'
done=root/'validate-longlive2-wan-norm.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
for name in ['finish-longlive2-wan-norm-evidence.exit','repeat-longlive2-wan-norm.exit','finish-longlive2-wan-norm-evidence-retry1.exit']:
    assert (root/name).read_text().strip()=='143','original waiting driver cancellation record missing'
subprocess.run(['git','-C',str(repo),'fetch',str(root/'longlive2-wan-norm-reviewed-final.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(repo),'merge','--ff-only','FETCH_HEAD'],check=True)
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
(root/'artifacts/longlive2/repeat-source.txt').write_text(head+'\n')
# Device implementation, dispatcher, and benchmark match the NCU-measured source.
subprocess.run(['git','-C',str(repo),'diff','--exit-code','a98b0875c4',head,'--','python/sglang/kernels','test/registered/kernels/benchmark',':(exclude)python/sglang/kernels/ops/diffusion/norm/wan_norm_silu_post.py'],check=True)
module_path='python/sglang/kernels/ops/diffusion/norm/wan_norm_silu_post.py'
old_module=subprocess.check_output(['git','-C',str(repo),'show','a98b0875c4:'+module_path],text=True)
assert (repo/module_path).read_text().replace('def _wan_norm_silu_post(', 'def _run(').replace('    _wan_norm_silu_post(', '    _run(')==old_module
env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'8','PYTHONPATH':str(repo/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1'}
def run(cmd,label):
    with (root/f'{label}.log').open('x') as out:p=subprocess.run(cmd,cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT)
    (root/f'{label}.exit').write_text(str(p.returncode));assert p.returncode==0,label
with (root/'gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    run(['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_wan_norm_silu_post.py','python/sglang/multimodal_gen/test/unit/test_vae_fast_path_gate.py','python/sglang/multimodal_gen/test/unit/test_diffusion_import_isolation.py'],'longlive2-norm-final-tests-retry2')
run(['python',str(root/'profile/longlive2-wan-norm-post-h200/analysis/analyze_ncu.py')],'longlive2-norm-ncu-analysis-retry2')
for mode in ('t2v','i2v'):
    baseline=json.loads((root/'artifacts/longlive2'/f'{mode}-baseline-a1'/'result.json').read_text())
    for repeat in (3,4):
        for arm in ('a1','b1','b2','a2'):
            label=f'{mode}-norm-r{repeat}-{arm}'
            source=root/'baseline' if arm.startswith('a') else repo
            run(['python','-u',str(root/'run-longlive2.py'),'--repo',str(source),'--mode',mode,'--label',label],'longlive2-'+label)
            record=json.loads((root/'artifacts/longlive2'/label/'result.json').read_text())
            assert record['output_sha256']==baseline['output_sha256'],label
    label=f'{mode}-norm-final-profile'
    run(['python','-u',str(root/'run-longlive2.py'),'--repo',str(repo),'--mode',mode,'--label',label,'--profile','--all-stages'],'longlive2-'+label)
    label=f'{mode}-norm-final-high-eager'
    run(['python','-u',str(root/'run-longlive2.py'),'--repo',str(repo),'--mode',mode,'--label',label,'--quality','high'],'longlive2-'+label)
(root/'repeat-longlive2-wan-norm-final.exit').write_text('0')
run(['python',str(root/'audit-longlive2-wan-norm.py')],'longlive2-norm-audit-retry2')
