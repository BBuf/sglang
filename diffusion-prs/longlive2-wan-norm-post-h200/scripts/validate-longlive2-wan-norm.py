"""Repeated native A/B, complete profiles, and quality/applicability evidence."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/campaign');repo=root/'candidate-longlive2-wan-norm'
done=root/'run-longlive2-norm-admission.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
for mode in ('t2v','i2v'):
    base=json.loads((root/'artifacts/longlive2'/f'{mode}-baseline-a1'/'result.json').read_text())
    candidate=json.loads((root/'artifacts/longlive2'/f'{mode}-norm-candidate-a1'/'result.json').read_text())
    assert not candidate['error'] and candidate['output_sha256']==base['output_sha256']
subprocess.run(['git','-C',str(repo),'fetch',str(root/'longlive2-wan-norm-benchmark.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(repo),'merge','--ff-only','FETCH_HEAD'],check=True)
env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'8','PYTHONPATH':str(repo/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1'}

def run(mode,label,source,flags=(),disabled=False):
    log=root/f'longlive2-{label}.log'
    with log.open('x') as out:
        result=subprocess.run(['python','-u',str(root/'run-longlive2.py'),'--repo',str(source),'--mode',mode,'--label',label,*flags],env=env,stdout=out,stderr=subprocess.STDOUT)
    (root/f'longlive2-{label}.exit').write_text(str(result.returncode))
    record=json.loads((root/'artifacts/longlive2'/label/'result.json').read_text())
    if result.returncode:
        assert disabled and record['error'] and '[diffusion bcg] disabled' in log.read_text().lower(),label
    else:
        assert not disabled,label+' unexpectedly enabled BCG'
        if record['quality']=='lossless':
            baseline=json.loads((root/'artifacts/longlive2'/f'{mode}-baseline-a1'/'result.json').read_text())
            assert record['output_sha256']==baseline['output_sha256'],label
    return record

for mode in ('t2v','i2v'):
    for repeat in (1,2):
        for arm in ('a1','b1','b2','a2'):
            run(mode,f'{mode}-norm-r{repeat}-{arm}',root/'baseline' if arm.startswith('a') else repo)
    run(mode,f'{mode}-norm-profile',repo,['--profile','--all-stages'])
    trace=next((root/'artifacts/longlive2'/f'{mode}-norm-profile'/'traces').glob('*full_stages*.trace.json.gz'))
    subprocess.run(['python',str(root/'analyze-longlive2-norm-trace.py'),str(trace)],check=True)
    rows=[]
    for arm,source in [('baseline',root/'baseline'),('candidate',repo)]:
        eager=f'{mode}-norm-r2-'+('a1' if arm=='baseline' else 'b1')
        rows.append(json.loads((root/'artifacts/longlive2'/eager/'result.json').read_text()))
        if arm=='baseline':
            rows.append(json.loads((root/'artifacts/longlive2'/f'{mode}-bcg-applicability'/'result.json').read_text()))
        else:
            rows.append(run(mode,f'{mode}-norm-{arm}-lossless-bcg',source,['--bcg'],True))
        rows.append(run(mode,f'{mode}-norm-{arm}-high-eager',source,['--quality','high']))
        rows.append(run(mode,f'{mode}-norm-{arm}-high-bcg',source,['--quality','high','--bcg'],True))
    (root/'artifacts/longlive2'/f'{mode}-norm-quality-matrix.json').write_text(json.dumps(rows,indent=2))
with (root/'gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (root/'longlive2-norm-marker.log').open('x') as out:
        p=subprocess.run(['python','test/registered/kernels/benchmark/diffusion/bench_wan_norm_silu_post.py'],cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT)
    (root/'longlive2-norm-marker.exit').write_text(str(p.returncode));assert p.returncode==0
with (root/'longlive2-norm-ncu.log').open('x') as out:
    p=subprocess.run(['bash',str(root/'profile/longlive2-wan-norm-post-h200/collect.sh')],env=env,stdout=out,stderr=subprocess.STDOUT)
(root/'longlive2-norm-ncu.exit').write_text(str(p.returncode));assert p.returncode==0
