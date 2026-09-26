import importlib.util
import os
spec = importlib.util.spec_from_file_location("hunyuan_integrated", os.environ["CANDIDATE_SOURCE"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def run(op, kwargs):
    assert op == "hunyuan_qkv_rope_pack"
    return module.hunyuan_qkv_rope_pack(**kwargs)
def prepare_benchmark(op, kwargs):
    assert op == "hunyuan_qkv_rope_pack"
    return module.hunyuan_qkv_rope_pack, (), kwargs
