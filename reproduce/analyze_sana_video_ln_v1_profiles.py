"""Attribute one full native CFG iteration by CPU launch correlation, not timestamp overlap."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = []
for arm, folder in [('main', 'sana-video-native-qualification1-sgl'), ('candidate', 'sana-video-ln-v1-qualification1-sgl')]:
    trace = next((root/'evidence/profiles'/folder).glob('*.gz'))
    events = json.loads(gzip.decompress(trace.read_bytes()))['traceEvents']
    step = next(event for event in events if event.get('name') == 'ProfilerStep#3')
    start, end = step['ts'], step['ts']+step['dur']
    launches = {event['args']['correlation']: event for event in events
                if event.get('cat') in ['cuda_runtime', 'cuda_driver']
                and 'correlation' in event.get('args', {}) and 'LaunchKernel' in event.get('name', '')}
    kernels = [event for event in events if event.get('cat') == 'kernel'
               and event.get('args', {}).get('correlation') in launches
               and start <= launches[event['args']['correlation']]['ts'] < end]
    groups = collections.defaultdict(lambda: [0, 0.0])
    for kernel in kernels:
        groups[kernel['name']][0] += 1
        groups[kernel['name']][1] += kernel['dur']/1000
    counts = dict(fused_ln=sum(count for name, (count, _) in groups.items() if '_layernorm_modulate_kernel' in name),
                  aten_ln=sum(count for name, (count, _) in groups.items() if 'layer_norm_kernel' in name),
                  fused_rope=sum(count for name, (count, _) in groups.items() if 'interleaved_rope_fp64_kernel' in name),
                  fused_conv_silu=sum(count for name, (count, _) in groups.items() if '_bias_silu_kernel' in name),
                  fused_conv_glu=sum(count for name, (count, _) in groups.items() if '_bias_glu_kernel' in name))
    rows.append(dict(arm=arm, trace=str(trace.relative_to(root)), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        step='ProfilerStep#3', kernel_count=len(kernels), kernel_sum_ms=sum(kernel['dur'] for kernel in kernels)/1000,
        gpu_span_ms=(max(kernel['ts']+kernel['dur'] for kernel in kernels)-min(kernel['ts'] for kernel in kernels))/1000,
        counts=counts, kernels=[dict(name=name, count=count, sum_ms=duration) for name, (count, duration) in sorted(groups.items(), key=lambda item:-item[1][1])]))
assert rows[0]['counts']['aten_ln'] == rows[1]['counts']['fused_ln'] == 82
assert rows[0]['counts']['fused_ln'] == rows[1]['counts']['aten_ln'] == 0
for row in rows:
    assert all(row['counts'][name] == 40 for name in ['fused_rope', 'fused_conv_silu', 'fused_conv_glu'])
result = dict(method='All cuda_runtime/cuda_driver kernel launches in CPU ProfilerStep#3 joined to GPU kernels by correlation ID; captures complete launch work despite CPU/GPU asynchronous timestamps.',
    diagnostic_only=True, performance_admitted=False, actual_fused_path_verified=True, rows=rows,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out = root/'evidence/preflight/sana-video-ln-v1-profile1.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2)+'\n')
for row in rows:
    print(json.dumps({key:value for key,value in row.items() if key!='kernels'}))
