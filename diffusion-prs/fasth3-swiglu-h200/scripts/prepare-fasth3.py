"""Prepare the pinned native FastH3 export and overlay in an isolated cache."""
import importlib.util
import json
import os
from pathlib import Path
import shutil

root=Path('/campaign')
assert (root/'close-sensenova-cycle.exit').read_text().strip()=='0'
assert not (root/'model-caches/sensenova-u1-cycle').exists()
assert not (root/'model-caches/vdn-h3-cycle').exists()
helper=root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper)
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
cache_root=root/'model-caches'
cache=cache_root/'fasth3-cycle'
if not cache.exists():
    cache=bench._prepare_model_cache(cache_root,'fasth3','cycle')
else:
    assert (cache_root/bench.MODEL_CACHE_MARKER).is_file()
env=bench._model_cache_env(cache)|{
    'XDG_CACHE_HOME':str(cache/'xdg'),'SGLANG_DIFFUSION_CACHE_ROOT':str(cache/'sglang'),
    'HF_XET_CHUNK_CACHE_SIZE_BYTES':'0','CUDA_VISIBLE_DEVICES':'0,1',
    'FLASHINFER_DISABLE_VERSION_CHECK':'1','SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1',
}
os.environ.update(env)
(root/'artifacts/fasth3-cache-env.json').write_text(json.dumps(env,indent=2))
assert shutil.disk_usage(root).free>170*1024**3
from huggingface_hub import snapshot_download
requests=[('FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree','5ea076f35b84da4c3c82217112fa733d8eea2ae1'),
          ('kevin-mi/FastH3-4step-Preview-overlay','f769cb8001dae335089de7b250364335bc7cb183')]
resolved=[]
for repo,revision in requests:
    print('Downloading',repo,revision,flush=True)
    path=snapshot_download(repo,revision=revision,max_workers=4)
    ref=Path(path).parent.parent/'refs/main'
    ref.parent.mkdir(parents=True,exist_ok=True)
    ref.write_text(revision)
    resolved.append(dict(repo=repo,revision=revision,snapshot=path))
    (root/'artifacts/fasth3-snapshots.json').write_text(json.dumps(resolved,indent=2))
print('Pinned FastH3 checkpoint ready',flush=True)
