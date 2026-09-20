"""Final native ABBA, production-shape kernels, profiles and quality/BCG probes."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
root=Path('/campaign');repo=root/'candidate-cosmos3-super-final';base=root/'baseline-super-current'
done=root/'validate-cosmos3-super-prototype.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
subprocess.run(['git','-C',str(root/'baseline-current'),'fetch',str(root/'cosmos3-super-final.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(root/'baseline-current'),'worktree','add','--detach',str(repo),'1a7e397700ad6f6d7c1bb5af719411c9937bec1f'],check=True)
assert not subprocess.check_output(['git','-C',str(repo),'diff','86c40bd29225f21f986aff26101f1c47f58695b8','HEAD','--','python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py'],text=True)
env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'8','PYTHONPATH':str(repo/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1'}
with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 selected=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 for _ in range(35):
  procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if not any(line.startswith(selected) for line in procs.splitlines()):break
  time.sleep(1)
 else:raise RuntimeError('Assigned GPU busy; no process killed')
 for label,cmd in [('model-tests',['python','-m','pytest','-q','python/sglang/multimodal_gen/test/unit/test_cosmos3.py']),('kernel-tests',['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_rope.py','-k','qknorm_rope_pack_kv or cosmos3_super_t2i']),('marker',['python','test/registered/kernels/benchmark/diffusion/bench_cosmos3_super_qk_pack.py'])]:
  with (root/f'cosmos3-super-final-{label}.log').open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
  (root/f'cosmos3-super-final-{label}.exit').write_text(str(r.returncode));assert r.returncode==0,label

def run(label,source,flags=(),disabled=False):
 with (root/f'cosmos3-super-t2i-{label}.log').open('x') as log:
  r=subprocess.run(['python','-u',str(root/'run-cosmos3-super-t2i.py'),'--repo',str(source),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
 (root/f'cosmos3-super-t2i-{label}.exit').write_text(str(r.returncode))
 result=json.loads((root/'artifacts/cosmos3-super-t2i'/label/'result.json').read_text())
 if disabled:
  assert r.returncode and result['error'] and '[diffusion bcg] disabled' in (root/f'cosmos3-super-t2i-{label}.log').read_text().lower(),label
 else:
  assert r.returncode==0 and not result['error'],label
  if result['quality']=='lossless':assert result['output_sha256']==['cfd2735b1e5d3b64a888ccc631f48d173e95c32857e8cbebd9e80fde3bc90d5e'],label
 return result

for repeat in (1,2):
 for arm in ('a1','b1','b2','a2'):run(f'final-r{repeat}-{arm}',base if arm.startswith('a') else repo)
for arm,source in [('baseline',base),('candidate',repo)]:
 run(f'final-{arm}-profile',source,['--profile'])
 run(f'final-{arm}-high-eager',source,['--quality','high'])
 run(f'final-{arm}-high-bcg',source,['--quality','high','--bcg'],True)
run('final-candidate-lossless-bcg',repo,['--bcg'],True)
print('All fixed native comparisons and applicability checks completed')
