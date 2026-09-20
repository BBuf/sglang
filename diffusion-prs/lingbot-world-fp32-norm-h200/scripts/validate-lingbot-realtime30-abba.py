"""Two fixed ABBA groups for the native thirty-chunk WebSocket workload, retaining first-chunk setup."""
import json
from pathlib import Path
import subprocess

root = Path('/campaign')
while not (root/'audit-lingbot-realtime-abba.exit').exists():
    __import__('time').sleep(5)
assert (root/'audit-lingbot-realtime-abba.exit').read_text().strip() == '0'
ref = None
index = 0
for repeat in (1, 2):
    for arm in ('a1', 'b1', 'b2', 'a2'):
        label = f'realtime30-r{repeat}-{arm}'
        repo = root/('baseline-current' if arm.startswith('a') else 'candidate-lingbot-current')
        with (root/f'lingbot-{label}.log').open('x') as log:
            result = subprocess.run(['python', '-u', str(root/'run-lingbot-realtime.py'),
                '--repo', str(repo), '--label', label, '--port', str(24200+index),
                '--master-port', str(24300+index), '--chunks', '30'], stdout=log, stderr=subprocess.STDOUT)
        (root/f'lingbot-{label}.exit').write_text(str(result.returncode))
        assert result.returncode == 0, label
        record = json.loads((root/'artifacts/lingbot-world-v2'/label/'result.json').read_text())
        if ref is None: ref = record
        assert record['chunks'] == 30 and record['frame_count'] == 357
        assert record['raw_frame_sha256'] == ref['raw_frame_sha256'], label
        assert record['output_sha256'] == ref['output_sha256'], label
        print(label, record['scheduler_forward_s'], record['client_saved_s'], flush=True)
        index += 1
print('Both native thirty-chunk ABBA groups completed; all raw frames and MP4s exact')
