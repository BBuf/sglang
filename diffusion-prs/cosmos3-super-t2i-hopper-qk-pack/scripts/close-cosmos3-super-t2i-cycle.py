"""Close reviewed Super T2I evidence and delete only this model's owned cache."""
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/campaign');art=root/'artifacts/cosmos3-super-t2i'
done=root/'audit-cosmos3-super-replication-retry2.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
e=json.loads((art/'final-evidence.json').read_text())
assert e['qualified'] and e['candidate']=='1a7e397700ad6f6d7c1bb5af719411c9937bec1f'
assert all(row.get('quality_pass',True) for row in e['outputs'])
assert (art/'media-visually-reviewed.txt').is_file()
review=json.loads((art/'media-visually-reviewed.txt').read_text())
assert review['comparison_sha256']==hashlib.sha256((art/'output-comparison/image-comparison.png').read_bytes()).hexdigest()
with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True)
 selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
 for _ in range(35):
  processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()):break
  time.sleep(1)
 else:raise RuntimeError('Assigned GPU busy; no process killed')
 os.environ['CUDA_VISIBLE_DEVICES']='0,1'
 import torch
 environment=dict(baseline=e['baseline'],candidate=e['candidate'],checkpoint='nvidia/Cosmos3-Super-Text2Image',
  checkpoint_revision='daf3d374804be4c512c2135568a7cb95d4341d79',gpu_inventory=inventory,used_gpu_indices=[0,1],
  dtype='bfloat16',quality='lossless',mode='eager',tp=2,sp=1,torch=torch.__version__,torch_git=torch.version.git_version,
  cuda=torch.version.cuda,python=subprocess.check_output(['python','--version'],text=True).strip(),
  versions={k:importlib.metadata.version(k) for k in ('triton','transformers','diffusers','sglang-kernel')},
  residency='manual: DiT and VAE resident, no torch.compile; same guardrail disable on both arms',
  numerical_contract='Reuse existing rounded BF16 QKNorm/RoPE/KV-pack; no CUDA arithmetic changes')
 (art/'environment.json').write_text(json.dumps(environment,indent=2))
 helper=root/'baseline-super-current/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
 spec=importlib.util.spec_from_file_location('campaign_bench',helper);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
 record=bench._cleanup_model_cache(root/'model-caches',root/'model-caches/cosmos3-super-t2i-cycle',
  root/'artifacts/cache-cleanup.jsonl','cosmos3-super-t2i','cycle',
  'Three fixed native ABBA groups, paired profiles, exact lossless outputs, high quality, real BCG applicability, production kernel tests/markers and manually reviewed images completed')
 assert record['after']==dict(file_count=0,weight_file_count=0,total_bytes=0)
 assert not (root/'model-caches/cosmos3-super-t2i-cycle').exists()
 (art/'cache-cleanup.json').write_text(json.dumps(record,indent=2))
 print(json.dumps(record,indent=2))
