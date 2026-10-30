"""Diagnose real source-function tensor lifetimes; never admit model performance."""
import ast
import datetime
import gc
import gzip
import hashlib
import inspect
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path

ROOT = Path('/task')
plan_path = Path(sys.argv[1])
plan = json.loads(plan_path.read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['script_sha256']
assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == plan['physical_gpu_uuid']
for name, expected in plan['trigger_files_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
trigger = json.loads((ROOT / plan['trigger_admission']).read_text())
assert trigger['primary_passed'] and not trigger['cleanup_passed'] and not trigger['passed']
destination = ROOT / 'outputs/diagnostics' / plan['id']
destination.mkdir(exist_ok=False, parents=True)
source_path = 'python/sglang/multimodal_gen/runtime/models/dits/joy_image.py'
sources, functions, trees = {}, {}, {}
for label, spec in plan['sources'].items():
    repo = ROOT / spec['repo']
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == spec['head']
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
    raw = (repo / source_path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec['file_sha256']
    text = raw.decode()
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_joy_image_mlp')
    sources[label] = ast.get_source_segment(text, node) + '\n'
    tree.body.remove(node)
    trees[label] = ast.dump(tree, include_attributes=False)
    (destination / (label + '-function.py')).write_text(sources[label])
assert trees['original'] == trees['current'], 'Other runtime changes need separate analysis'

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.models.dits import joy_image
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import ensure_distributed_env_defaults

assert Path(inspect.getfile(joy_image)).resolve() == (ROOT / plan['sources']['current']['repo'] / source_path).resolve()
assert torch.cuda.get_device_capability() == (12, 0)
for label, source in sources.items():
    namespace = vars(joy_image).copy()
    exec(compile(source, label + '-exact-source.py', 'exec'), namespace)
    functions[label] = namespace['_joy_image_mlp']
assert inspect.getsource(joy_image._joy_image_mlp) == sources['current']
if not model_parallel_is_initialized():
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

records = []
state = dict(status='running', purpose=plan['purpose'], plan=plan, records=records,
    python=sys.version, torch=torch.__version__, actual_device=str(torch.cuda.get_device_properties(0)),
    full_model_admission=False, timing_admission=False)
status_path = destination / 'results.json'
try:
    with torch.inference_mode():
        torch.manual_seed(plan['seed'])
        mlp = MLP(4096, 16384).to(device='cuda', dtype=torch.bfloat16)
        for parameter in mlp.parameters():
            parameter.normal_(0, 0.01)
        for tokens in plan['tokens']:
            base = torch.randn(1, tokens, 4096, device='cuda', dtype=torch.bfloat16)
            for label in ('original', 'current'):
                for _ in range(plan['warmup_calls_per_source']):
                    output = functions[label](mlp, base.clone(), True)
                    del output
            torch.cuda.synchronize()
            reference = None
            for index, label in enumerate(plan['order']):
                observed, holder = [], []

                def temporary(base_tensor=base):
                    value = base_tensor.clone()
                    holder.append(weakref.ref(value))
                    return value

                def before_output_projection(module, inputs):
                    observed.append(dict(input_still_alive=holder[0]() is not None,
                        allocated_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved(),
                        hidden_shape=list(inputs[0].shape)))

                hook = mlp.fc_out.register_forward_pre_hook(before_output_projection)
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                try:
                    output = functions[label](mlp, temporary(), True)
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated()
                    assert len(observed) == 1 and holder[0]() is None
                    cpu = output.cpu()
                    if reference is None:
                        reference = cpu.clone()
                    exact = torch.equal(cpu, reference)
                    assert exact
                    output_sha = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
                    del output
                finally:
                    hook.remove()
                # Shape profiling can retain tensors. Lifetime/memory observations above
                # deliberately come from the separate call without Torch Profiler.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                        profile_memory=True) as profiler:
                    with torch.profiler.record_function(label + '_real_MLP_temporary_input'):
                        profiled_output = functions[label](mlp, base.clone(), True)
                    torch.cuda.synchronize()
                assert torch.equal(profiled_output.cpu(), reference)
                del profiled_output
                try:
                    trace = destination / f'{tokens}-{index:02d}-{label}.trace.json'
                    profiler.export_chrome_trace(str(trace))
                    compressed = trace.with_suffix(trace.suffix + '.gz')
                    compressed.write_bytes(gzip.compress(trace.read_bytes(), mtime=0))
                    trace.unlink()
                    records.append(dict(tokens=tokens, index=index, source=label,
                        output_sha256=output_sha, output_exact=True,
                        before_allocated_bytes=before, peak_allocated_bytes=peak,
                        peak_increment_bytes=peak-before, output_projection_entry=observed[0],
                        lifetime_and_memory_observation_has_no_torch_profiler=True,
                        input_dead_after_return=True, trace=str(compressed.relative_to(ROOT)),
                        trace_sha256=hashlib.sha256(compressed.read_bytes()).hexdigest()))
                finally:
                    del cpu, profiler, temporary
                    gc.collect()
                torch.cuda.synchronize()
                status_path.write_text(json.dumps(state, indent=2) + '\n')
            del base, reference
        state['status'] = 'complete_causal_review_pending'
except Exception as exc:
    state.update(status='failed_preserve_outputs', error=repr(exc))
    raise
finally:
    state['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status_path.write_text(json.dumps(state, indent=2) + '\n')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
print(json.dumps(dict(status=state['status'], records=len(records), full_model_admission=False)))
