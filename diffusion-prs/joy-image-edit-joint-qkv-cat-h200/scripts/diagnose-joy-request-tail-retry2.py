"""Separate diagnostic worktrees; coarse request-tail timing, never E2E evidence."""
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
done = root / 'validate-joy-persistent-retry1.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
instrument = '''

# Campaign-only diagnostics. Never included in the upstream optimization branch.
import functools as _campaign_functools
import json as _campaign_json
import gc as _campaign_gc

_campaign_gc_starts = {}
def _campaign_gc_callback(phase, info):
    generation = info['generation']
    if phase == 'start':
        _campaign_gc_starts[generation] = time.perf_counter()
    elif generation in _campaign_gc_starts:
        duration = time.perf_counter() - _campaign_gc_starts.pop(generation)
        if duration > .01:
            print('CAMPAIGN_GC ' + _campaign_json.dumps(dict(pid=os.getpid(),
                generation=generation, seconds=duration, collected=info['collected'],
                monotonic=time.monotonic())), flush=True)
_campaign_gc.callbacks.append(_campaign_gc_callback)

def _campaign_tail_wrap(name, function):
    @_campaign_functools.wraps(function)
    def measured(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            print('CAMPAIGN_TAIL ' + _campaign_json.dumps(dict(
                name=name, pid=os.getpid(), rank=os.environ.get('LOCAL_RANK'),
                seconds=time.perf_counter()-start, monotonic=time.monotonic())), flush=True)
    return measured

for _campaign_name in ['_materialize_output_transport', '_record_output_peak_memory', '_record_replica_peak_memory']:
    setattr(GPUWorker, _campaign_name, _campaign_tail_wrap(_campaign_name, getattr(GPUWorker, _campaign_name)))
torch.cuda.empty_cache = _campaign_tail_wrap('torch.cuda.empty_cache', torch.cuda.empty_cache)
capture_memory_snapshot = _campaign_tail_wrap('capture_memory_snapshot', capture_memory_snapshot)
for _campaign_name in ['dump_benchmark_report', 'log_request_summary']:
    setattr(PerformanceLogger, _campaign_name, staticmethod(_campaign_tail_wrap(_campaign_name, getattr(PerformanceLogger, _campaign_name))))
'''
runner = (root / 'run-joy-persistent-retry1.py').read_text()
runner = runner.replace("/ 'persistent' / args.label", "/ 'tail-diagnostic' / args.label")
runner = runner.replace('for index in range(6):', 'for index in range(12):')
runner = runner.replace('measured_requests=5', 'measured_requests=11')
(root / 'run-joy-tail-diagnostic.py').write_text(runner)
for arm, sha in [('baseline', '80da4432d085ed4d6166ef643d9fd2b829dbb0c5'), ('candidate', '4811f4d52aa2586412f699b3bb84ed184d76250a')]:
    repo = root / f'diagnostic-joy-tail-{arm}'
    if not repo.exists():
        subprocess.run(['git', '-C', str(root / 'baseline-super-current'), 'worktree', 'add', '--detach', str(repo), sha], check=True)
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == sha
    source = repo / 'python/sglang/multimodal_gen/runtime/managers/gpu_worker.py'
    if 'def _campaign_tail_wrap(' not in source.read_text():
        source.write_text(source.read_text() + instrument)
    else:
        assert source.read_text().endswith(instrument)
    subprocess.run(['git', '-C', str(repo), 'add', str(source)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'core.hooksPath=/dev/null', '-c', 'user.name=BBuf', '-c', 'user.email=1182563586@qq.com', 'commit', '-m', 'Campaign diagnostics: time native request-tail operations'], check=True)
    with (root / f'joy-image-edit-tail-diagnostic-{arm}.log').open('x') as log:
        code = subprocess.run(['python', '-u', str(root / 'run-joy-tail-diagnostic.py'), '--repo', str(repo), '--label', arm], stdout=log, stderr=subprocess.STDOUT).returncode
    (root / f'joy-image-edit-tail-diagnostic-{arm}.exit').write_text(str(code))
    assert code == 0, arm
print('Diagnostic outputs and timings preserved separately; no qualification from instrumented runs', flush=True)
