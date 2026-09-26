import importlib.util
import os
spec = importlib.util.spec_from_file_location("ltx_integrated", os.environ["CANDIDATE_SOURCE"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def run(op, kwargs):
    assert op == "ltx2_ada_values9"
    return module.ltx2_ada_values9(**kwargs)
def prepare_benchmark(op, kwargs):
    assert op == "ltx2_ada_values9"
    return module.ltx2_ada_values9, (), kwargs
