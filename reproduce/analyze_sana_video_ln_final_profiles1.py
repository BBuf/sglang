"""Attribute a complete CFG iteration to actual eager or graph kernel launches."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = []
folders = [
    ('main_eager', 'sana-video-native-qualification1-sgl'),
    ('v1_eager', 'sana-video-ln-v1-qualification1-sgl'),
    ('final_eager', 'sana-video-ln-final-compatibility1-01-final_eager'),
    ('main_bcg', 'sana-video-ln-final-compatibility1-02-main_bcg'),
    ('final_bcg', 'sana-video-ln-final-compatibility1-03-final_bcg'),
]
for arm, folder in folders:
    traces = list((root/'evidence/profiles'/folder).glob('*.gz'))
    assert len(traces) == 1, (arm, traces)
    trace = traces[0]
    events = json.loads(gzip.decompress(trace.read_bytes()))['traceEvents']
    steps = [event for event in events if event.get('name') == 'ProfilerStep#3'
             and event.get('cat') == 'user_annotation']
    assert len(steps) == 1, (arm, steps)
    step = steps[0]
    start, end = step['ts'], step['ts']+step['dur']
    launches = {event['args']['correlation']: event for event in events
                if event.get('cat') in ['cuda_runtime', 'cuda_driver']
                and 'correlation' in event.get('args', {})
                and any(name in event.get('name', '') for name in ['LaunchKernel', 'GraphLaunch'])
                and start <= event['ts'] < end}
    kernels = [event for event in events if event.get('cat') == 'kernel'
               and event.get('args', {}).get('correlation') in launches]
    assert kernels, arm
    groups = collections.defaultdict(lambda: [0, 0.0])
    for kernel in kernels:
        groups[kernel['name']][0] += 1
        groups[kernel['name']][1] += kernel['dur']/1000
    counts = dict(fused_ln=sum(count for name, (count, _) in groups.items() if '_layernorm_modulate_kernel' in name),
                  aten_ln=sum(count for name, (count, _) in groups.items() if 'layer_norm_kernel' in name),
                  fused_rope=sum(count for name, (count, _) in groups.items() if 'interleaved_rope_fp64_kernel' in name),
                  fused_conv_silu=sum(count for name, (count, _) in groups.items() if '_bias_silu_kernel' in name),
                  fused_conv_glu=sum(count for name, (count, _) in groups.items() if '_bias_glu_kernel' in name))
    graph_launches = sum('GraphLaunch' in event['name'] for event in launches.values())
    graph_kernels = [kernel for kernel in kernels if 'GraphLaunch' in launches[kernel['args']['correlation']]['name']]
    assert counts['fused_ln'] == (0 if arm == 'main_eager' else 82), (arm, counts)
    assert counts['aten_ln'] == (82 if arm == 'main_eager' else 0), (arm, counts)
    assert all(counts[name] == 40 for name in ['fused_rope', 'fused_conv_silu', 'fused_conv_glu']), (arm, counts)
    if arm.endswith('_bcg'):
        assert graph_launches and graph_kernels, arm
        assert sum('_layernorm_modulate_kernel' in kernel['name'] for kernel in graph_kernels) == 82, arm
    else:
        assert not graph_launches, arm
    rows.append(dict(arm=arm, trace=str(trace.relative_to(root)), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                     step='CPU user_annotation ProfilerStep#3', launch_calls=len(launches), graph_launches=graph_launches,
                     graph_kernel_count=len(graph_kernels), kernel_count=len(kernels), counts=counts,
                     kernel_sum_ms=sum(kernel['dur'] for kernel in kernels)/1000,
                     gpu_span_ms=(max(kernel['ts']+kernel['dur'] for kernel in kernels)-min(kernel['ts'] for kernel in kernels))/1000,
                     kernels=[dict(name=name, count=count, sum_ms=duration)
                              for name, (count, duration) in sorted(groups.items(), key=lambda item:-item[1][1])]))
result = dict(passed=True, diagnostic_only=True, performance_admitted=False,
              method='CPU user_annotation ProfilerStep#3 launches joined to GPU kernels by CUDA correlation ID; includes CUDA Graph launches and all their correlated graph-node kernels. GPU timestamp overlap alone is not used.',
              rows=rows, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out = root/'evidence/preflight/sana-video-ln-final-profile1.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2)+'\n')
for row in rows:
    print(json.dumps({key:value for key,value in row.items() if key != 'kernels'}))
