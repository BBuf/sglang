"""Screen the lossless QK/RoPE candidate with saved output and native profiling."""
import json
from pathlib import Path
import subprocess

root=Path('/campaign');repo=root/'candidate-klein-qk-integration'
subprocess.run(['git','-C',str(root/'baseline-super-current'),'fetch',str(root/'klein-qk-integration.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(root/'baseline-super-current'),'worktree','add','--detach',str(repo),'a95d30db76ba3292ef7bb144310ec6d95fd0e90d'],check=True)
reference=json.loads((root/'artifacts/flux2-klein-base-4b/baseline-a1/result.json').read_text())
for label,flags in [('qk-candidate-b0',[]),('qk-candidate-profile',['--profile'])]:
 with (root/f'flux2-klein-base-4b-{label}.log').open('x') as log:
  proc=subprocess.run(['python','-u',str(root/'run-flux2-klein-base-4b.py'),'--repo',str(repo),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
 (root/f'flux2-klein-base-4b-{label}.exit').write_text(str(proc.returncode))
 assert proc.returncode==0,label
 result=json.loads((root/'artifacts/flux2-klein-base-4b'/label/'result.json').read_text())
 assert not result['error']
 assert result['output_sha256']==reference['output_sha256'],(label,result['output_sha256'],reference['output_sha256'])
 print(label,json.dumps(result),flush=True)
 if flags:
  trace=Path(result['native_profile_traces'][0]['path'])
  with (trace.parent.parent/'inspect.txt').open('w') as out:
   subprocess.run(['python',str(root/'inspect-flux2-klein-trace.py'),str(trace)],stdout=out,check=True)
  data=json.loads((trace.parent/'forward3-evidence.json').read_text())
  fused=[r for r in data['top_kernels'] if r['name']=='_flux2_strided_qknorm_rope_kernel']
  assert len(fused)==1 and fused[0]['count']==20,fused
  print('Twenty fused QK/RoPE launches in complete native forward',fused,flush=True)
