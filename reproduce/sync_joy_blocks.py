"""Copy completed planned blocks without mixing server and benchmark metadata."""
import argparse
import json
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('plan',type=Path)
p.add_argument('indices',type=int,nargs='+')
a=p.parse_args();plan=json.loads(a.plan.read_text())
root=Path(__file__).resolve().parents[1]
remote='SGL-RTX6000-01:/data/xyzhang/diffusion-rtx6000-20261026/'
for i in a.indices:
 config=plan['order'][i];name=f'{plan["id"]}-{i:02d}-{config}'
 arm='omni' if config.startswith('omni') else 'sgl'
 pairs=[('outputs/benchmarks/'+name,'benchmarks'),('block-runs/'+name,'block-runs'),
        ('outputs/qualification/'+arm+'-joy-'+name,'qualification')]
 if config in plan.get('quality_references', {}):
  pairs.append(('outputs/qualification/'+arm+'-joy-'+name+'-quality.json','qualification'))
 for source,target in pairs:
  dest=root/'evidence'/target;dest.mkdir(exist_ok=True)
  subprocess.run(['scp','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
                  '-r',remote+source,str(dest)+'/'],check=True)
 print(name+' archived',flush=True)
