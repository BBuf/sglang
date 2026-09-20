"""Download pinned Cosmos3 Super Text2Image only after audited LingBot World V2 cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import hashlib
import urllib.request

root=Path('/campaign')
assert not (root/'model-caches/lingbot-world-v2-cycle').exists(), 'Finish LingBot World V2 and audit cleanup first'
cleanup=json.loads((root/'artifacts/lingbot-world-v2/cache-cleanup.json').read_text())
assert cleanup['after']==dict(file_count=0,weight_file_count=0,total_bytes=0)
helper=root/'baseline-current/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper)
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
cache_root=root/'model-caches'
cache=cache_root/'cosmos3-super-t2i-cycle'
if not cache.exists():
    cache=bench._prepare_model_cache(cache_root,'cosmos3-super-t2i','cycle')
else:
    assert (cache_root/bench.MODEL_CACHE_MARKER).is_file()
env=bench._model_cache_env(cache)|{
    'XDG_CACHE_HOME':str(cache/'xdg'),
    'SGLANG_DIFFUSION_CACHE_ROOT':str(cache/'sglang'),
    'HF_XET_CHUNK_CACHE_SIZE_BYTES':'0',
    'CUDA_VISIBLE_DEVICES':'0,1',
    'FLASHINFER_DISABLE_VERSION_CHECK':'1',
    'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1',
}
os.environ.update(env)
(root/'artifacts/cosmos3-super-t2i-cache-env.json').write_text(json.dumps(env,indent=2))
assert shutil.disk_usage(root).free>200*1024**3
from huggingface_hub import snapshot_download
repo='nvidia/Cosmos3-Super-Text2Image'
revision='daf3d374804be4c512c2135568a7cb95d4341d79'
path=snapshot_download(repo,revision=revision,max_workers=4,ignore_patterns=["assets/*","images/*"])
ref=Path(path).parent.parent/'refs/main'
ref.parent.mkdir(parents=True,exist_ok=True)
ref.write_text(revision)
(root/'artifacts/cosmos3-super-t2i-snapshot.json').write_text(json.dumps(dict(repo=repo,revision=revision,snapshot=path),indent=2))
print('Pinned Cosmos3 Super Text2Image checkpoint ready:',path,flush=True)
