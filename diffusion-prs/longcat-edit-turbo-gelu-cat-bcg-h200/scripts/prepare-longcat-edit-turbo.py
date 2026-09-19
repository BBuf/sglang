"""Download pinned LongCat Image Edit Turbo only after the FastH3 cache is removed."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import hashlib
import urllib.request

root=Path('/campaign')
assert not (root/'model-caches/fasth3-cycle').exists(), 'Finish FastH3 and audit cleanup first'
helper=root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper)
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
cache_root=root/'model-caches'
cache=cache_root/'longcat-image-edit-turbo-cycle'
if not cache.exists():
    cache=bench._prepare_model_cache(cache_root,'longcat-image-edit-turbo','cycle')
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
(root/'artifacts/longcat-edit-turbo-cache-env.json').write_text(json.dumps(env,indent=2))
assert shutil.disk_usage(root).free>50*1024**3
from huggingface_hub import snapshot_download
repo='meituan-longcat/LongCat-Image-Edit-Turbo'
revision='6a7262de5549f0bf0ec54c08ef7d283ef41f3214'
path=snapshot_download(repo,revision=revision,max_workers=4)
ref=Path(path).parent.parent/'refs/main'
ref.parent.mkdir(parents=True,exist_ok=True)
ref.write_text(revision)
(root/'artifacts/longcat-edit-turbo-snapshot.json').write_text(json.dumps(dict(repo=repo,revision=revision,snapshot=path),indent=2))
print('Pinned LongCat Image Edit Turbo checkpoint ready:',path,flush=True)
media=root/'artifacts/input-media/longcat-edit-input.jpg'
media.parent.mkdir(parents=True,exist_ok=True)
url='https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg'
if not media.exists():
    with urllib.request.urlopen(url,timeout=60) as stream:
        media.write_bytes(stream.read())
media.with_suffix('.json').write_text(json.dumps(dict(url=url,path=str(media),sha256=hashlib.sha256(media.read_bytes()).hexdigest()),indent=2))
