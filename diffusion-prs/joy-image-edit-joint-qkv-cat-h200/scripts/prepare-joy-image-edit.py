"""Download pinned JoyAI image edit only after audited FLUX.2 Klein base 4B cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import hashlib
import urllib.request

root=Path('/campaign')
assert not (root/'model-caches/flux2-klein-base-4b-cycle').exists(), 'Finish FLUX.2 Klein base 4B and audit cleanup first'
cleanup=json.loads((root/'artifacts/flux2-klein-base-4b/cache-cleanup.json').read_text())
assert cleanup['after']==dict(file_count=0,weight_file_count=0,total_bytes=0)
helper=root/'baseline-super-current/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper)
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
cache_root=root/'model-caches'
cache=cache_root/'joy-image-edit-cycle'
if not cache.exists():
    cache=bench._prepare_model_cache(cache_root,'joy-image-edit','cycle')
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
(root/'artifacts/joy-image-edit-cache-env.json').write_text(json.dumps(env,indent=2))
assert shutil.disk_usage(root).free>110*1024**3
from huggingface_hub import snapshot_download
repo='jdopensource/JoyAI-Image-Edit-Diffusers'
revision='4b41fb25d961f37668750178ccbb380da326201c'
path=snapshot_download(repo,revision=revision,max_workers=4)
ref=Path(path).parent.parent/'refs/main'
ref.parent.mkdir(parents=True,exist_ok=True)
ref.write_text(revision)
(root/'artifacts/joy-image-edit-snapshot.json').write_text(json.dumps(dict(repo=repo,revision=revision,snapshot=path),indent=2))
print('Pinned JoyAI image edit checkpoint ready:',path,flush=True)
