"""Compare complete GPU ProfilerStep scopes; reject raw partial-window counts."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
report = {'scope': 'Rank 0 only. Requested two profiled timesteps; traces contain three complete GPU ProfilerStep intervals plus an unscoped partial tail. Compare matching complete intervals, never whole-trace counts or summed ranks.', 'arms': {}}


def family(name):
    if name == 'nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_bias_TNT':
        return 'GEMM 256x128 TNT'
    if name == 'nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN':
        return 'GEMM 192x192 TNN'
    if 'FlashAttnFwdSm90' in name:
        return 'FlashAttention 3'
    if 'GeluCUDAKernelImpl' in name:
        return 'GELU'
    if 'fused_qknorm_rope_warp' in name:
        return 'Fused QKNorm RoPE'
    if 'nccl' in name.lower():
        return 'NCCL'
    return 'Other'


for arm in ['a1', 'b1']:
    cell = root / 'results' / f'qwen-image-layered-cfg2-pr3-lossless-profile-{arm}'
    traces = list((cell / 'profiler').glob('*.gz'))
    assert len(traces) == 1
    trace = json.load(gzip.open(traces[0], 'rt'))
    events = trace['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    scopes = sorted([e for e in events if e.get('cat') == 'gpu_user_annotation' and e['name'].startswith('ProfilerStep#')], key=lambda e: e['ts'])
    steps = []
    for scope in scopes:
        selected = [e for e in kernels if e['pid'] == scope['pid'] and scope['ts'] <= e['ts'] < scope['ts'] + scope['dur']]
        groups = defaultdict(lambda: {'count': 0, 'gpu_ms': 0.0})
        for event in selected:
            row = groups[family(event['name'])]
            row['count'] += 1
            row['gpu_ms'] += event['dur'] / 1000
        steps.append({'scope': scope['name'], 'range_ms': scope['dur'] / 1000, 'kernel_count': len(selected), 'cumulative_kernel_ms': sum(e['dur'] for e in selected) / 1000, 'families': dict(groups)})
    assert len(steps) == 3
    assert len({s['kernel_count'] for s in steps}) == 1
    report['arms'][arm] = {'trace': str(traces[0]), 'raw_kernel_count': len(kernels), 'selected_kernel_count': sum(s['kernel_count'] for s in steps), 'unscoped_kernel_count': len(kernels) - sum(s['kernel_count'] for s in steps), 'steps': steps}

source = report['arms']['a1']['steps']
candidate = report['arms']['b1']['steps']
assert [s['scope'] for s in source] == [s['scope'] for s in candidate]
report['per_step_mean'] = {}
for arm, steps in [('source', source), ('candidate_rank0', candidate)]:
    means = {key: statistics.mean(s[key] for s in steps) for key in ['range_ms', 'kernel_count', 'cumulative_kernel_ms']}
    means['families'] = {name: {key: statistics.mean(s['families'].get(name, {}).get(key, 0) for s in steps) for key in ['count', 'gpu_ms']} for name in sorted(set().union(*(s['families'] for s in steps)))}
    report['per_step_mean'][arm] = means
out = root / 'results/qwen-layered-pr3-complete-step-profile-comparison.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['per_step_mean'], indent=2))
