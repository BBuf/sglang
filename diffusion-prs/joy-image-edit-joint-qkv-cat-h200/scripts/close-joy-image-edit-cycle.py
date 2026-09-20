"""Delete only the campaign-owned Joy checkpoint after completed media audit."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import time

root=Path('/campaign');art=root/'artifacts/joy-image-edit'
assert (root/'audit-joy-client-save.exit').read_text().strip()=='0'
assert (root/'diagnose-joy-request-tail-retry2.exit').read_text().strip()=='0'
assert json.loads((art/'final-evidence.json').read_text())['qualified']
assert json.loads((art/'media-visually-reviewed.json').read_text())['reviewed']
with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
 selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
 for _ in range(35):
  processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()):break
  time.sleep(1)
 else:raise RuntimeError('Assigned GPU busy; no process killed')
 helper=root/'baseline-super-current/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
 spec=importlib.util.spec_from_file_location('campaign_bench',helper);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
 record=bench._cleanup_model_cache(root/'model-caches',root/'model-caches/joy-image-edit-cycle',root/'artifacts/cache-cleanup.jsonl','joy-image-edit','cycle','Qualifying native eager comparisons, paired profile/NCU, exact images, high/disabled BCG checks and visual audit complete')
 assert record['after']==dict(file_count=0,weight_file_count=0,total_bytes=0)
 assert not (root/'model-caches/joy-image-edit-cycle').exists()
 (art/'cache-cleanup.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
