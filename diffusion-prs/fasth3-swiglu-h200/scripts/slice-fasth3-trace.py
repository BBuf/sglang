"""Keep one complete synchronized FastH3 iteration and attribute SwiGLU launches."""
import bisect
import collections
import gzip
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
with gzip.open(source, 'rt') as stream:
    data = json.load(stream)
events = data['traceEvents']
models = sorted((e for e in events if e.get('name') == 'nn.Module: MiniMaxH3DiTModel_0'), key=lambda e: e['ts'])
assert len(models) >= 3, len(models)
# StageProfiler synchronizes both sides of each iteration with synchronized
# stage profiling enabled. Consecutive model starts include the complete GPU tail.
start, end = models[1]['ts'], models[2]['ts']
crossing = [e for e in events if e.get('cat') == 'kernel' and
            any(e['ts'] < t < e['ts'] + e.get('dur', 0) for t in (start, end))]
assert not crossing, crossing[:3]
selected = []
for e in events:
    ts = e.get('ts')
    if e.get('ph') == 'M':
        selected.append(e)
    elif ts is not None and start <= ts < end:
        if e.get('ph') == 'X' and ts + e.get('dur', 0) > end:
            e = e | {'dur': end-ts, 'args': e.get('args', {}) | {'campaign_clipped_cpu_span': True}}
        selected.append(e)
    elif ts is not None and e.get('ph') == 'X' and ts < start < ts+e.get('dur', 0) and e.get('cat') != 'kernel':
        selected.append(e | {'ts': start, 'dur': min(end, ts+e['dur'])-start,
                             'args': e.get('args', {}) | {'campaign_clipped_cpu_span': True}})
scopes = sorted((e for e in selected if e.get('cat') == 'python_function' and
                 'minimax_h3.py' in e.get('name', '') and e['name'].endswith(': _silu_mul')), key=lambda e: e['ts'])
assert len(scopes) == 50, len(scopes)
starts = [e['ts'] for e in scopes]
launches = {e.get('args', {}).get('correlation'): e for e in selected
            if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'LaunchKernel' in e.get('name', '')}
attributed = collections.defaultdict(lambda: [0, 0.0])
for e in selected:
    if e.get('cat') != 'kernel':
        continue
    launch = launches.get(e.get('args', {}).get('correlation'))
    if launch is None:
        continue
    i = bisect.bisect_right(starts, launch['ts'])-1
    if i >= 0 and launch['ts'] < scopes[i]['ts']+scopes[i]['dur']:
        attributed[e['name']][0] += 1
        attributed[e['name']][1] += e['dur']/1000
target = source.with_name('denoise-step2.trace.json.gz')
with gzip.open(target, 'wt', compresslevel=1) as stream:
    json.dump(data | {'traceEvents': selected}, stream, separators=(',', ':'))
report = dict(source=str(source), slice=str(target), source_event_count=len(events),
              slice_event_count=len(selected), start_us=start, end_us=end,
              window_ms=(end-start)/1000, crossing_kernel_count=0,
              boundary_rule='Second recorded MiniMaxH3DiTModel start to third; stage synchronization enabled',
              swiglu_calls=len(scopes), swiglu_kernel_count=sum(v[0] for v in attributed.values()),
              swiglu_gpu_ms=sum(v[1] for v in attributed.values()),
              swiglu_kernels=[dict(name=k, count=v[0], total_ms=v[1]) for k, v in attributed.items()])
target.with_name('denoise-step2-evidence.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
