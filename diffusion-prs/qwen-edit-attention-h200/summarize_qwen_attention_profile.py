"""Deduplicate overlapping stage annotations before comparing profile evidence."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', choices=['torch211', 'torch213'], required=True)
args = parser.parse_args()
root = Path('/data/goals/sglang-h200-diffusion-20260907')
prefix = {'torch211': 'ion8-attention-final-gpu6', 'torch213': 'ion9-attention-pr-torch213'}[args.runtime]


def totals(events):
    result = defaultdict(lambda: dict(count=0, gpu_ms=0.0, bytes=0))
    for event in events:
        item = result[event['name']]
        item['count'] += 1
        item['gpu_ms'] += event['dur'] / 1000
        item['bytes'] += event.get('args', {}).get('bytes', 0)
    return [dict(name=name, **value) for name, value in sorted(result.items(), key=lambda x: x[1]['gpu_ms'], reverse=True)]


report = dict(runtime=args.runtime, scope='One H200, rank0, full two-step stage captures. GPU stage intervals are unioned per name/device; each event is counted once. Profile durations include instrumentation and are not model E2E benchmark results.', arms={})
for arm in ['a1', 'b1']:
    cell = root / 'results' / f'qwen-edit-base-{prefix}-lossless-profile-2steps-{arm}'
    traces = list((cell / 'profiler').glob('*.gz'))
    assert len(traces) == 1
    path = traces[0]
    with gzip.open(path, 'rt') as stream:
        data = json.load(stream)
    events = data['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    copies = [e for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    scopes = [e for e in events if e.get('cat') == 'gpu_user_annotation' and e['name'].startswith('STAGE ')]
    cpu_stages = [dict(name=e['name'], duration_ms=e['dur']/1000) for e in events if e.get('cat') == 'user_annotation' and e['name'].startswith('STAGE ')]
    grouped = defaultdict(list)
    for event in scopes:
        grouped[(event['name'], event['pid'])].append((event['ts'], event['ts'] + event['dur']))
    stages = []
    for (name, pid), intervals in grouped.items():
        merged = []
        for begin, end in sorted(intervals):
            if merged and begin <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([begin, end])

        def inside(event):
            return event['pid'] == pid and any(begin <= event['ts'] < end for begin, end in merged)

        selected = [e for e in kernels if inside(e)]
        selected_copies = [e for e in copies if inside(e)]
        crossing = [e for e in selected if not any(begin <= e['ts'] and e['ts'] + e['dur'] <= end + 0.01 for begin, end in merged)]
        stages.append(dict(name=name, gpu_pid=pid, annotation_count=len(intervals), union_intervals=merged,
                           union_duration_ms=sum(end-begin for begin, end in merged)/1000,
                           kernel_count=len(selected), cumulative_kernel_ms=sum(e['dur'] for e in selected)/1000,
                           boundary_crossing_kernel_count=len(crossing), kernels=totals(selected), copies=totals(selected_copies)))
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    label = f'{prefix}-lossless-profile-2steps-{arm}'
    perf = json.loads((cell / f'qwen-edit-base_{label}.json').read_text())
    report['arms'][arm] = dict(trace=str(path), trace_sha256=sha.hexdigest(), trace_bytes=path.stat().st_size,
                               commit=perf['commit_hash'], measured_steps=len(perf['denoise_steps_ms']),
                               raw_kernel_count=len(kernels), stages=stages, cpu_stages=cpu_stages)
    del data, events, kernels, copies, scopes
report['comparison'] = []
a = {(s['name'], s['gpu_pid']): s for s in report['arms']['a1']['stages']}
b = {(s['name'], s['gpu_pid']): s for s in report['arms']['b1']['stages']}
assert a.keys() == b.keys()
for key in a:
    report['comparison'].append(dict(name=key[0], gpu_pid=key[1],
        source_kernels=a[key]['kernel_count'], candidate_kernels=b[key]['kernel_count'],
        removed_kernels=a[key]['kernel_count']-b[key]['kernel_count'],
        source_cumulative_kernel_ms=a[key]['cumulative_kernel_ms'], candidate_cumulative_kernel_ms=b[key]['cumulative_kernel_ms']))
out = root / 'results' / f'qwen-edit-attention-{args.runtime}-union-profile.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(steps={k:v['measured_steps'] for k,v in report['arms'].items()}, comparison=report['comparison']), indent=2), flush=True)
