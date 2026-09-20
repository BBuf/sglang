"""Two fixed ABBA groups for the native ten-chunk WebSocket workload."""
import json
from pathlib import Path
import subprocess

root = Path('/campaign')
assert (root/'repeat-lingbot-current-retry1.exit').read_text().strip() == '0'
assert (root/'audit-lingbot-current.exit').read_text().strip() == '0'
ref = json.loads((root/'artifacts/lingbot-world-v2/current-realtime-baseline/result.json').read_text())
index = 0
for repeat in (1, 2):
    for arm in ('a1', 'b1', 'b2', 'a2'):
        label = f'realtime-r{repeat}-{arm}'
        repo = root/('baseline-current' if arm.startswith('a') else 'candidate-lingbot-current')
        with (root/f'lingbot-{label}.log').open('x') as log:
            result = subprocess.run(['python', '-u', str(root/'run-lingbot-realtime.py'),
                '--repo', str(repo), '--label', label, '--port', str(24000+index),
                '--master-port', str(24100+index)], stdout=log, stderr=subprocess.STDOUT)
        (root/f'lingbot-{label}.exit').write_text(str(result.returncode))
        assert result.returncode == 0, label
        record = json.loads((root/'artifacts/lingbot-world-v2'/label/'result.json').read_text())
        assert record['raw_frame_sha256'] == ref['raw_frame_sha256'], label
        assert record['output_sha256'] == ref['output_sha256'], label
        print(label, record['scheduler_forward_s'], record['client_saved_s'], flush=True)
        index += 1
print('Both native ten-chunk ABBA groups completed; all raw frames and MP4s exact')
