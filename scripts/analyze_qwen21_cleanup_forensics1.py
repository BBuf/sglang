"""Compare complete matched steps without promoting profile timing to primary."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

from analyze_joy_gelu_profiles import aggregate, inside

ROOT = Path(__file__).resolve().parents[1]
RUN = 'qwen21-cleanup-1024-forensics1'
audit_path = ROOT/'evidence/preflight'/(RUN+'-audit.json')
audit = json.loads(audit_path.read_text())
assert audit['all_raw_audits_pass'] and audit['failed_primary_remains_failed']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def union_duration(events):
    end, total = -float('inf'), 0
    for event in sorted(events, key=lambda e:e['ts']):
        finish = event['ts'] + event['dur']
        total += max(0, finish-max(end, event['ts']))
        end = max(end, finish)
    return total/1000


cases = {}
for task in ['t2i', 'edit']:
    arms = {}
    for config in ['original', 'current']:
        spec = audit['tasks'][config][task]
        path = ROOT/spec['profile_trace']
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec['profile_sha256']
        events = json.loads(gzip.decompress(raw))['traceEvents']
        rows = {}
        # #1 is the boundary step; compare all three interior steps consistently.
        for step in [2, 3, 4]:
            name = f'ProfilerStep#{step}'
            anchors = {cat:[e for e in events if e.get('cat') == cat and e.get('name') == name]
                       for cat in ['user_annotation', 'gpu_user_annotation']}
            assert all(len(v) == 1 for v in anchors.values())
            ca, ga = anchors['user_annotation'][0], anchors['gpu_user_annotation'][0]
            assert ca['args']['External id'] == ga['args']['External id']
            cpu = sorted([e for e in events if e.get('cat') == 'cpu_op' and inside(e, ca)], key=lambda e:e['ts'])
            gpu = sorted([e for e in events if e.get('cat') == 'kernel' and inside(e, ga)], key=lambda e:e['ts'])
            runtime = [e for e in events if e.get('cat') == 'cuda_runtime' and inside(e, ca)]
            shapes = collections.Counter(json.dumps(e.get('args', {}).get('Input Dims'))
                                         for e in cpu if e['name'] == 'aten::mm')
            cpu_sequence = [(e['name'], e.get('args', {}).get('Input Dims'), e.get('args', {}).get('Input type')) for e in cpu]
            gpu_sequence = [(e['name'], {k:e.get('args', {}).get(k) for k in ['grid', 'block', 'registers per thread', 'shared memory']}) for e in gpu]
            runtime_counts = dict(collections.Counter(e['name'] for e in runtime))
            mm_shapes = [dict(input_dims=json.loads(s), calls=n) for s,n in sorted(shapes.items())]
            packed = sum(x['calls'] for x in mm_shapes if x['input_dims'] == [[4096,4096],[4096,24576]])
            silu = sum(e['name'] == '_packed_silu_mul_kernel' for e in gpu)
            rows[str(step)] = dict(cpu_step_ms=ca['dur']/1000, gpu_step_ms=ga['dur']/1000,
                kernel_count=len(gpu), kernel_sum_ms=sum(e['dur'] for e in gpu)/1000,
                kernel_union_ms=union_duration(gpu), gpu_span_uncovered_by_kernel_events_ms=ga['dur']/1000-union_duration(gpu),
                cuda_runtime_union_ms=union_duration(runtime), runtime_counts=runtime_counts,
                kernels=aggregate(gpu), cpu_op_count=len(cpu), cpu_op_counts=dict(collections.Counter(e['name'] for e in cpu)),
                cpu_sequence_sha256=fingerprint(cpu_sequence), gpu_sequence_sha256=fingerprint(gpu_sequence),
                mm_shapes=mm_shapes, packed_gemms=packed, packed_silu_kernels=silu,
                expected_packed_dispatch=packed == silu == 32, expected_kernel_count=len(gpu) == 384,
                graph_launches=runtime_counts.get('cudaGraphLaunch', 0))
        arms[config] = dict(source_sha=audit['source_heads'][config], trace=spec['profile_trace'],
            trace_sha256=spec['profile_sha256'], output_sha256=spec['profile_output_sha256'], steps=rows,
            synchronized_denoise_seconds=spec['denoise_seconds'], synchronized_denoise_p50_seconds=spec['denoise_p50_seconds'],
            peak_sampled_memory_bytes=spec['peak_sampled_memory_bytes'])
    comparisons = {}
    for step in ['2', '3', '4']:
        a,b = arms['original']['steps'][step], arms['current']['steps'][step]
        comparisons[step] = {key:a[key] == b[key] for key in ['kernel_count','gpu_sequence_sha256','mm_shapes','cpu_op_count','cpu_op_counts','cpu_sequence_sha256','runtime_counts']}
    cases[task] = dict(arms=arms, comparisons=comparisons,
        equal_profile_output=arms['original']['output_sha256'] == arms['current']['output_sha256'],
        equal_peak_sampled_memory=arms['original']['peak_sampled_memory_bytes'] == arms['current']['peak_sampled_memory_bytes'])
result = dict(raw_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(), primary_timing=False,
    failed_primary_remains_failed=True, steps=[2,3,4], cases=cases,
    all_gpu_sequences_match=all(v['gpu_sequence_sha256'] for c in cases.values() for v in c['comparisons'].values()),
    all_cpu_sequences_match=all(v['cpu_sequence_sha256'] for c in cases.values() for v in c['comparisons'].values()),
    all_expected_packed_dispatch=all(s['expected_packed_dispatch'] and s['expected_kernel_count'] and not s['graph_launches'] for c in cases.values() for a in c['arms'].values() for s in a['steps'].values()),
    caveat='One fresh process per source, two tasks, three matched interior profile steps per trace. Profile durations and SYNC1 denoise are descriptive and excluded from SYNC0 primary admission. GPU gaps include all activity not covered by kernel events, not pure CPU launch delay. CPU operator traces do not isolate Python argument binding. Identical dispatch does not prove identical performance or explain failed primary cleanup; its two CIs include one. No source repair is justified by equality alone.')
out = ROOT/'evidence/profiles'/(RUN+'-profile-review.json')
assert not out.exists()
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'cases'}))
for task, case in cases.items():
    print(json.dumps(dict(task=task, comparisons=case['comparisons'], arms={arm:{'gpu_step_ms':[v['gpu_step_ms'] for v in x['steps'].values()], 'cpu_step_ms':[v['cpu_step_ms'] for v in x['steps'].values()], 'kernel_counts':[v['kernel_count'] for v in x['steps'].values()], 'packed_gemms':[v['packed_gemms'] for v in x['steps'].values()]} for arm,x in case['arms'].items()})))
