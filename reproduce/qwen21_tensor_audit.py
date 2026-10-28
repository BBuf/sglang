"""Observation-only native stage snapshots, outside compiled transformer blocks."""
import hashlib
import json
import os
from pathlib import Path

import torch

_active = None
_counts = {}


def begin(prompt):
    global _active
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    allowed = []
    for task in ['t2i', 'edit']:
        contract = json.loads(Path(f'/task/inputs/qwen21-base-{task}-workloads.json').read_text())
        allowed.extend(row['prompt'] for row in contract['prompts'])
    if prompt not in allowed:
        _active = None
        return
    key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    index = _counts.get(key, 0)
    _counts[key] = index + 1
    _active = Path(os.environ['QWEN21_DIAGNOSTIC_DIR']) / f'{key}-{index:02d}'
    _active.mkdir(parents=True, exist_ok=False)
    (_active / 'metadata.json').write_text(json.dumps(dict(
        prompt=prompt, diagnostic_only=True, torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        autocast_cuda=torch.is_autocast_enabled('cuda'),
    ), indent=2)+'\n')


def snapshot(name, **values):
    if _active is None:
        return
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cpu(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(type(value))
    path = _active / (name+'.pt')
    assert not path.exists(), path
    torch.save(cpu(values), path)


def step(index, phase, **values):
    if index in (0, 1, 25, 49):
        snapshot(f'step-{index:02d}-{phase}', **values)
