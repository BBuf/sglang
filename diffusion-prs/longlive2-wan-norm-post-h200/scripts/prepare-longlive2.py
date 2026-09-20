"""Download pinned LongLive2 only after the MiniMax H3 cache is removed."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import hashlib
import urllib.request

root=Path('/campaign')
assert not (root/'model-caches/minimax-h3-cycle').exists(), 'Finish MiniMax H3 and audit cleanup first'
helper=root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper)
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
cache_root=root/'model-caches'
cache=cache_root/'longlive2-cycle'
if not cache.exists():
    cache=bench._prepare_model_cache(cache_root,'longlive2','cycle')
else:
    assert (cache_root/bench.MODEL_CACHE_MARKER).is_file()
env=bench._model_cache_env(cache)|{
    'XDG_CACHE_HOME':str(cache/'xdg'),
    'SGLANG_DIFFUSION_CACHE_ROOT':str(cache/'sglang'),
    'HF_XET_CHUNK_CACHE_SIZE_BYTES':'0',
    'CUDA_VISIBLE_DEVICES':'0',
    'FLASHINFER_DISABLE_VERSION_CHECK':'1',
    'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1',
}
os.environ.update(env)
(root/'artifacts/longlive2-cache-env.json').write_text(json.dumps(env,indent=2))
assert shutil.disk_usage(root).free>45*1024**3
from huggingface_hub import snapshot_download
repo='Rabinovich/LongLive-2.0-5B-Diffusers'
revision='37664a64cfc38cdc5fabe88576961ef10646e4f1'
path=snapshot_download(repo,revision=revision,max_workers=4)
ref=Path(path).parent.parent/'refs/main'
ref.parent.mkdir(parents=True,exist_ok=True)
ref.write_text(revision)
(root/'artifacts/longlive2-snapshot.json').write_text(json.dumps(dict(repo=repo,revision=revision,snapshot=path),indent=2))
print('Pinned LongLive2 checkpoint ready:',path,flush=True)
