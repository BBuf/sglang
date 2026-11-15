"""Attribute the frozen native Nomas trace to CPU stages and GPU launches."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

root = Path('/task')
folder = root / 'runs/nomas-49-sgl-profile05'
status = json.loads((folder / 'status.json').read_text())
assert status['status'] == 'raw_complete' and status['gpu_cleanup'] == 'idle_verified'
requests = json.loads((folder / 'outputs/results.json').read_text())
assert all(x['matches_reference'] and not x['benchmark'] for x in requests)
trace, = (folder / 'profiles').glob('*.trace.json.gz')
with gzip.open(trace, 'rt') as f:
    events = json.load(f)['traceEvents']
stages = [e for e in events if e.get('cat') == 'user_annotation' and e['name'].startswith('STAGE ')]
cpu_categories = ('cpu_op', 'user_annotation', 'cuda_runtime', 'cuda_driver')
cpu_events = [e for e in events if e.get('ph') == 'X' and e.get('cat') in cpu_categories]
external = {e['args']['External id']: e for e in cpu_events if 'External id' in e.get('args', {})}
launches = {e['args']['correlation']: e for e in cpu_events if 'correlation' in e.get('args', {})}
groups = collections.defaultdict(list)
unresolved = []
for e in events:
    if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset') or e.get('ph') != 'X':
        continue
    args = e.get('args', {})
    owner = launches.get(args.get('correlation')) or external.get(args.get('External id'))
    matches = [s for s in stages if owner and s['tid'] == owner['tid'] and s['ts'] <= owner['ts'] < s['ts'] + s['dur']]
    assert len(matches) <= 1
    stage = matches[0]['name'] if matches else 'unresolved'
    groups[stage].append(e)
    if not matches:
        unresolved.append(dict(name=e['name'], args=args))


def union_us(es):
    end = -float('inf')
    total = 0
    for a, b in sorted((e['ts'], e['ts'] + e['dur']) for e in es):
        total += max(0, b - max(end, a))
        end = max(end, b)
    return total


def costs(es):
    sums = collections.Counter()
    counts = collections.Counter()
    for e in es:
        sums[e['name']] += e['dur']
        counts[e['name']] += 1
    return [dict(name=n, count=counts[n], sum_ms=d/1000) for n, d in sums.most_common()]


rows = []
for stage, es in groups.items():
    s = next((s for s in stages if s['name'] == stage), None)
    cpu = [e for e in cpu_events if s and e['tid'] == s['tid'] and s['ts'] <= e['ts'] < s['ts'] + s['dur']]
    ks = [e for e in es if e['cat'] == 'kernel']
    span = max(e['ts'] + e['dur'] for e in es) - min(e['ts'] for e in es)
    shapes = collections.Counter((e['name'], json.dumps(e.get('args', {}).get('Input Dims'))) for e in cpu if e['name'] in ('aten::mm', 'aten::addmm', 'aten::native_layer_norm', 'aten::mean', 'aten::cat', 'aten::copy_', 'aten::_scaled_dot_product_cudnn_attention'))
    rows.append(dict(stage=stage, cpu_stage_ms=s['dur']/1000 if s else None,
        gpu_span_ms=span/1000, gpu_union_ms=union_us(es)/1000, gpu_gap_ms=(span-union_us(es))/1000,
        kernel_count=len(ks), kernel_sum_ms=sum(e['dur'] for e in ks)/1000,
        kernels=costs(ks), cpu_ops=costs(cpu),
        transfer_count=sum(e['cat'] != 'kernel' for e in es),
        shapes=[dict(name=n, dims=json.loads(d), count=c) for (n,d),c in shapes.most_common()]))
summary = dict(benchmark=False, cross_framework_performance_admitted=False,
    trace=str(trace), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
    output_reference_matches=[x['matches_reference'] for x in requests],
    categories=dict(collections.Counter(e.get('cat', '') for e in events)),
    stages=rows, unresolved_count=len(unresolved), unresolved_samples=unresolved[:10],
    interpretation='Each GPU event assigned once by correlated CPU launch/External id within a STAGE interval. CPU inclusive op costs overlap. GPU span, union, sums and gaps are separate; profiler overhead prevents treating gaps as removable latency. Native main only; failed cross-framework quality remains excluded.')
out = root / 'evidence/profiles/nomas-49-sgl-profile05/raw-costs.json'
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('x') as f:
    json.dump(summary, f, indent=2)
    f.write('\n')
print(json.dumps([{k:r[k] for k in ('stage','kernel_count','kernel_sum_ms','gpu_union_ms','gpu_gap_ms')} for r in rows],indent=2))
