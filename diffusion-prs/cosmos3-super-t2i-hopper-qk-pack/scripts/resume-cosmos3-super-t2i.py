"""Complete missing repository assets, preserve failed admission, resume native evidence."""
import json
import os
from pathlib import Path
import subprocess
root=Path('/campaign')
assert (root/'advance-cosmos3-super-t2i.exit').read_text().strip()=='1'
assert (root/'prepare-cosmos3-super-t2i.exit').read_text().strip()=='0'
assert not (root/'model-caches/lingbot-world-v2-cycle').exists()
os.environ.update(json.loads((root/'artifacts/cosmos3-super-t2i-cache-env.json').read_text()))
os.environ.pop('HF_HUB_OFFLINE',None)
from huggingface_hub import snapshot_download
meta=json.loads((root/'artifacts/cosmos3-super-t2i-snapshot.json').read_text())
path=snapshot_download(meta['repo'],revision=meta['revision'],max_workers=4)
assert path==meta['snapshot']
(root/'artifacts/cosmos3-super-t2i/full-snapshot-completed.json').write_text(json.dumps(dict(repo=meta['repo'],revision=meta['revision'],reason='Pinned weights present; native offline snapshot validation also requires seven README asset files. Downloaded complete pinned snapshot, no model source change.'),indent=2))
for label, flags in [('baseline-a1-retry1',[]),('baseline-profile',['--profile','--all-stages']),('baseline-bcg-probe',['--bcg'])]:
 with (root/f'cosmos3-super-t2i-{label}.log').open('x') as log:
  result=subprocess.run(['python','-u',str(root/'run-cosmos3-super-t2i.py'),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
 (root/f'cosmos3-super-t2i-{label}.exit').write_text(str(result.returncode))
 if label!='baseline-bcg-probe':assert result.returncode==0,label
print('Native baseline admission, profile and BCG attempt completed')
