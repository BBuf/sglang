"""Aggregate raw dense-Qwen21 trace costs; never use profiled time as a benchmark."""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

root = Path('/task')
for run in sys.argv[1:]:
    assert Path(run).name == run
    folder = root/'runs'/run
    status = json.loads((folder/'status.json').read_text())
    assert status['status'] == 'raw_complete' and status['gpu_cleanup'] == 'idle_verified'
    requests = [r for r in json.loads((folder/'outputs/results.json').read_text()) if r['profiler']]
    traces = list((folder/'profiles').glob('*.trace.json.gz'))
    assert len(requests) == len(traces) == 2
    for request in requests:
        log = (Path(request['output']).parent/'server-request.log').read_text()
        matched = [p for p in traces if p.name in log]
        assert len(matched) == 1 and request['matches_reference']
        trace, = matched
        output = root/'evidence/profiles'/run/request['prompt_id']/'raw-costs.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        assert not output.exists()
        with gzip.open(trace, 'rt') as file:
            events = json.load(file)['traceEvents']
        kernels = collections.defaultdict(lambda: [0, 0.0])
        cpu = collections.defaultdict(lambda: [0, 0.0])
        intervals = []
        shapes = collections.Counter()
        categories = collections.Counter()
        for event in events:
            category = event.get('cat', '')
            categories[category] += 1
            if event.get('ph') != 'X' or 'dur' not in event:
                continue
            name = event.get('name', '')
            if category == 'kernel':
                kernels[name][0] += 1
                kernels[name][1] += event['dur']
                intervals.append((event['ts'], event['ts'] + event['dur']))
            if category in ('cpu_op', 'cuda_runtime', 'user_annotation'):
                cpu[name][0] += 1
                cpu[name][1] += event['dur']
                if name in ('aten::mm', 'aten::addmm', 'aten::_flash_attention_forward', 'aten::stack', 'aten::cat', 'aten::copy_', 'aten::zero_'):
                    shapes[(name, json.dumps(event.get('args', {}).get('Input Dims')))] += 1
        intervals.sort()
        covered = 0.0
        end = -float('inf')
        for start, finish in intervals:
            covered += max(0, finish - max(end, start))
            end = max(end, finish)
        span = intervals[-1][1] - intervals[0][0]
        rows = lambda data: [dict(name=name, count=value[0], total_ms=value[1] / 1000)
                             for name, value in sorted(data.items(), key=lambda item: -item[1][1])]
        summary = dict(run=run, prompt_id=request['prompt_id'], benchmark=False, trace=str(trace),
                       trace_sha256=hashlib.file_digest(trace.open('rb'), 'sha256').hexdigest(),
                       categories=categories, kernel_sum_ms=sum(v[1] for v in kernels.values()) / 1000,
                       kernel_interval_union_ms=covered / 1000, first_to_last_kernel_span_ms=span / 1000,
                       gap_ms=(span-covered) / 1000, kernels=rows(kernels), cpu_ops=rows(cpu),
                       shapes=[dict(name=key[0], input_dims=json.loads(key[1]), count=count)
                               for key, count in shapes.most_common()],
                       interpretation='Dense single-GPU model. Kernel union gaps include profiler overhead and all pipeline stages, not causal CPU launch latency. CPU op times overlap and cannot be summed as wall time. No MoE or collective labels inferred.')
        output.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({key: summary[key] for key in ['run', 'kernel_sum_ms', 'gap_ms']}), flush=True)
        del events
