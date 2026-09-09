"""Frozen-source FLUX BCG applicability, repeated E2E, and profile evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

root = Path('/data/goals/sglang-h200-diffusion-20260907')
revision = '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'
repos = {
    'a': (root / 'next-model-baseline', '482e9f257bb9d9ac57c2245c93768a96ec70edbe'),
    'b': (root / 'flux-bcg-probe-repo', '5e91bc39e7a421be979448661b1b5d93db35cb48'),
}
assert Path(sys.prefix) == root / 'envs/qwen-torch213'
assert torch.__version__ == '2.13.0+cu130'
for repo, commit in repos.values():
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
output = root / 'results/flux-ion9-bcg-validation-v1-controller.json'
assert not output.exists(), 'Inspect existing controller PID/state before restarting.'
cache = root / 'model-caches/flux-ion9-torch213-initial-lossless-eager'
checkpoint = cache / 'huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots' / revision
assert checkpoint.is_dir()
state = dict(pid=os.getpid(), started_at=time.time(), gpus='3,6', torch=torch.__version__,
             checkpoint=revision, repos={k: dict(path=str(v[0]), commit=v[1]) for k,v in repos.items()}, rows=[])
cells = [('full-probe-b', 'b', 'lossless', 50, False), ('high-probe-b', 'b', 'high', 2, False)]
for repeat in (1, 2):
    cells.extend((f'll-r{repeat}-{i}-{side}', side, 'lossless', 50, False)
                 for i,side in enumerate('abba', 1))
cells.extend((f'profile-{side}', side, 'lossless', 2, True) for side in 'ab')
try:
    for name, side, quality, steps, profile in cells:
        label = f'ion9-bcg-validation-v1-{name}'
        repo, commit = repos[side]
        command = [sys.executable, '-u', str(root/'history/run_resume_cell_pr.py'),
                   '--repo', str(repo), '--root', str(root), '--model', 'flux', '--label', label,
                   '--gpus', '3,6', '--quality', quality, '--timeout', '3600',
                   '--hf-token-file', '/tmp/h200-diffusion-hf-token',
                   '--local-model-path', str(checkpoint), '--seed-cache', str(cache/'huggingface'),
                   '--extra-arg=--master-port=29675',
                   '--extra-arg=--model-id=black-forest-labs/FLUX.1-dev',
                   f'--extra-arg=--revision={revision}', f'--extra-arg=--num-inference-steps={steps}']
        if side == 'b':
            command.append('--bcg')
        if profile:
            command.extend(['--profile', '--profile-all-stages'])
        state['current_cell'] = label
        output.write_text(json.dumps(state, indent=2)+'\n')
        print('START', label, flush=True)
        log = root/'resume-logs'/f'flux-{label}.log'
        with log.open('w') as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        cell = root/'results'/f'flux-{label}'
        result = json.loads((cell/'result.json').read_text())
        text = log.read_text(errors='replace')
        assert not any(x in text for x in ('Falling back to diffusers backend', 'Using diffusers backend', 'Loaded diffusers pipeline'))
        perf_file = cell/f'flux_{label}.json'
        perf = json.loads(perf_file.read_text()) if perf_file.exists() else None
        record = dict(label=label, side=side, quality=quality, steps=steps, profile=profile,
                      returncode=proc.returncode, result=result, actual_perf_commit=perf.get('commit_hash') if perf else None)
        state['rows'].append(record)
        output.write_text(json.dumps(state, indent=2)+'\n')
        if quality == 'high' and result.get('bcg_invalid_signals'):
            print('HIGH_BCG_INAPPLICABLE', result['bcg_invalid_signals'], flush=True)
            continue
        assert proc.returncode == 0 and not result.get('error'), record
        assert perf and perf['commit_hash'] == commit and len(perf['denoise_steps_ms']) == steps
        if side == 'b':
            assert result.get('bcg_capture_detected') and not result.get('bcg_invalid_signals')
        if quality == 'lossless':
            expected = ('a713777ad67b4e82915aef972612f243d6f4cbd8db9aebdfdff976b4c7d2c7d3' if steps == 50
                        else '538ad2eae92eb2c2554e1a4abad65a0551011b2e02759b263f03d2ed4b964990')
            assert result['output_sha256'] == [expected], record
        print('CELL', label, result.get('e2e_latency_s'), result.get('denoise_latency_s'), result.get('output_sha256'), flush=True)
    state['status'] = 'complete'
except BaseException as exc:
    state['status'] = 'failed'
    state['exception'] = repr(exc)
    raise
finally:
    state['finished_at'] = time.time()
    output.write_text(json.dumps(state, indent=2)+'\n')
