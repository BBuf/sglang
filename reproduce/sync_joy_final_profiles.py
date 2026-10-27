"""Archive one completed final-head profile pair with source/PNG/trace metadata."""
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('shape', choices=['square','wide'])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
remote = 'SGL-RTX6000-01:/data/xyzhang/diffusion-rtx6000-20261026/'
pairs = []
for arm in ('main','pr'):
    name = f'joy-gelu-final-{args.shape}-profile1-{arm}'
    pairs.extend([(f'block-runs/{name}','block-runs'),
                  (f'outputs/profiles/{name}','profiles'),
                  (f'outputs/qualification/sgl-{name}-high','qualification')])
pairs.append((f'outputs/profiles/joy-gelu-final-{args.shape}-matched-profile-analysis.json','profiles'))
for source,target in pairs:
    destination = root/'evidence'/target
    destination.mkdir(exist_ok=True)
    subprocess.run(['scp','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
                    '-r',remote+source,str(destination)+'/'],check=True)
    print(source+' archived',flush=True)
