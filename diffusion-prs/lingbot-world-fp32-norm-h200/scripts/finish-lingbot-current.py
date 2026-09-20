"""Finish native multi-chunk correctness and NCU after fixed CLI measurements."""
import json
from pathlib import Path
import subprocess
import time
root=Path('/campaign')
done=root/'validate-lingbot-current.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0',done
for arm,source in [('baseline','baseline-current'),('candidate','candidate-lingbot-current')]:
    label='current-realtime-'+arm
    with (root/f'lingbot-{label}.log').open('x') as out:
        result=subprocess.run(['python','-u',str(root/'run-lingbot-realtime.py'),'--repo',str(root/source),'--label',label],stdout=out,stderr=subprocess.STDOUT)
    (root/f'lingbot-{label}.exit').write_text(str(result.returncode))
    assert result.returncode==0,label
art=root/'artifacts/lingbot-world-v2'
a=json.loads((art/'current-realtime-baseline/result.json').read_text())
b=json.loads((art/'current-realtime-candidate/result.json').read_text())
assert a['raw_frame_sha256']==b['raw_frame_sha256']
assert a['output_sha256']==b['output_sha256']
with (root/'lingbot-current-ncu.log').open('x') as out:
    result=subprocess.run(['bash',str(root/'profile/lingbot-fp32-norm-h200/collect.sh')],stdout=out,stderr=subprocess.STDOUT)
(root/'lingbot-current-ncu.exit').write_text(str(result.returncode))
assert result.returncode==0,'NCU'
print('Native multi-chunk raw frames and MP4 are exact; NCU collected')
