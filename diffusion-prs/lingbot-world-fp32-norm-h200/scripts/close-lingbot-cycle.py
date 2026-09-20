"""Close the manually reviewed LingBot evidence cycle and remove only owned weights."""
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
art = root/'artifacts/lingbot-world-v2'
assert (root/'audit-lingbot-current.exit').read_text().strip() == '0'
assert (root/'audit-lingbot-realtime-abba.exit').read_text().strip() == '0'
assert (root/'audit-lingbot-realtime30-abba.exit').read_text().strip() == '0'
assert (root/'lingbot-current-ncu-analysis-final.exit').read_text().strip() == '0'
report = json.loads((art/'final-evidence.json').read_text())
assert report['qualified'] and all(report['long_realtime_groups'][f'r{i}']['qualified'] for i in (1, 2))
assert report['candidate'] == '89b3700dbc9146727cbb368b6859f5a18e79ae66'
assert all(r.get('quality_pass', True) for r in report['outputs'] if r['valid_request'])
assert report['realtime']['baseline']['raw_frame_sha256'] == report['realtime']['candidate']['raw_frame_sha256']
lock = (root/'gpu.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv,noheader,nounits'], text=True)
selected = {line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0', '1'}}
for _ in range(30):
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()): break
    time.sleep(1)
else: raise RuntimeError('Assigned GPU busy; no process was killed')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
environment = dict(baseline=report['baseline'], candidate=report['candidate'],
    checkpoint='robbyant/lingbot-world-v2-14b-causal-fast-diffusers',
    checkpoint_revision='59cccf49f2d2dd27418ae7a04b82b10868d455c2',
    gpu_inventory=inventory, used_gpu_indices=[0], dtype='bfloat16', quality='lossless',
    mode='eager', torch=torch.__version__, torch_git=torch.version.git_version,
    cuda=torch.version.cuda, python=subprocess.check_output(['python', '--version'], text=True).strip(),
    versions={k: importlib.metadata.version(k) for k in ('triton', 'transformers', 'diffusers', 'sglang-kernel')},
    ncu=subprocess.check_output(['ncu', '--version'], text=True),
    residency='manual: DiT/VAE resident; native text encoder CPU offload; no compile',
    input_sha256=hashlib.sha256((root/'artifacts/input-media/longlive2-cat.png').read_bytes()).hexdigest(),
    numerical_contract='BF16 input/output with FP32 LayerNorm statistics and mixed FP32 modulation; existing rounded camera kernel')
(art/'environment.json').write_text(json.dumps(environment, indent=2))
helper = root/'baseline-current/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec = importlib.util.spec_from_file_location('campaign_bench', helper)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
record = bench._cleanup_model_cache(root/'model-caches', root/'model-caches/lingbot-world-v2-cycle',
    root/'artifacts/cache-cleanup.jsonl', 'lingbot-world-v2', 'cycle',
    'Final source tests, paired profiles/NCU, two native thirty-chunk ABBA groups, earlier ten-chunk limitations, all four short-CLI groups and their client limitations, raw exactness and decoded video audit reviewed')
assert record['after'] == dict(file_count=0, weight_file_count=0, total_bytes=0)
assert not (root/'model-caches/lingbot-world-v2-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
