"""Additional predeclared ABBA because second group has baseline latency outliers."""
import json
from pathlib import Path
import subprocess
import time

root=Path('/campaign')
for name in ('validate-cosmos3-super-final','audit-cosmos3-super-final'):
 done=root/f'{name}.exit'
 while not done.exists():time.sleep(5)
 assert done.read_text().strip()=='0',name
for arm in ('a1','b1','b2','a2'):
 label=f'final-r3-{arm}';source=root/('baseline-super-current' if arm.startswith('a') else 'candidate-cosmos3-super-final')
 with (root/f'cosmos3-super-t2i-{label}.log').open('x') as log:
  result=subprocess.run(['python','-u',str(root/'run-cosmos3-super-t2i.py'),'--repo',str(source),'--label',label],stdout=log,stderr=subprocess.STDOUT)
 (root/f'cosmos3-super-t2i-{label}.exit').write_text(str(result.returncode));assert result.returncode==0,label
 evidence=json.loads((root/'artifacts/cosmos3-super-t2i'/label/'result.json').read_text())
 assert not evidence['error'] and evidence['output_sha256']==['cfd2735b1e5d3b64a888ccc631f48d173e95c32857e8cbebd9e80fde3bc90d5e']
print('Additional complete ABBA finished; all original rows retained')
