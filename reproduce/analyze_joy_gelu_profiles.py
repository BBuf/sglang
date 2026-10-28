"""Compare complete matching Joy profile steps; never use trace time as E2E."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def aggregate(events):
    groups = collections.defaultdict(lambda: [0, 0.0])
    for event in events:
        group = groups[event['name']]
        group[0] += 1
        group[1] += event['dur'] / 1000
    return [dict(name=name, calls=n, sum_ms=duration)
            for name, (n, duration) in sorted(groups.items(), key=lambda item: -item[1][1])]


def inside(event, anchor):
    return (event.get('ph') == 'X' and event['ts'] >= anchor['ts']
            and event['ts'] + event['dur'] <= anchor['ts'] + anchor['dur'] + 1)


def analyze(path, step):
    raw = path.read_bytes()
    events = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)['traceEvents']
    name = f'ProfilerStep#{step}'
    cpu_anchors = [e for e in events if e.get('cat') == 'user_annotation' and e['name'] == name]
    gpu_anchors = [e for e in events if e.get('cat') == 'gpu_user_annotation' and e['name'] == name]
    assert len(cpu_anchors) == len(gpu_anchors) == 1
    cpu_anchor, gpu_anchor = cpu_anchors[0], gpu_anchors[0]
    assert cpu_anchor['args']['External id'] == gpu_anchor['args']['External id']
    cpu = [e for e in events if e.get('cat') == 'cpu_op' and inside(e, cpu_anchor)]
    kernels = [e for e in events if e.get('cat') == 'kernel' and inside(e, gpu_anchor)]
    # CUDA External id attributes identify the CPU operator that launched a kernel.
    activation = [e for e in cpu if e['name'] == 'aten::_addmm_activation']
    activation_ids = {e['args']['External id'] for e in activation}
    activation_kernels = [e for e in kernels if e.get('args', {}).get('External id') in activation_ids]
    gelu_shapes = collections.Counter(json.dumps(e.get('args', {}).get('Input Dims'))
                                     for e in cpu if e['name'] == 'aten::gelu')
    activation_shapes = collections.Counter(json.dumps(e.get('args', {}).get('Input Dims')) for e in activation)
    custom_ops = [e for e in cpu if 'diffusion_fused_linear_gelu_tanh' in e['name']]
    return dict(trace=str(path), sha256=hashlib.sha256(raw).hexdigest(), step=name,
                cpu_annotation_ms=cpu_anchor['dur']/1000, gpu_annotation_ms=gpu_anchor['dur']/1000,
                kernel_count=len(kernels), kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,
                cuda_graph_launch_count=sum(e.get('name') == 'cudaGraphLaunch' and inside(e, cpu_anchor) for e in events),
                gelu_cpu_calls=sum(gelu_shapes.values()),
                gelu_input_shapes=[dict(input_dims=json.loads(shape), calls=n) for shape,n in gelu_shapes.items()],
                fused_custom_op_calls=len(custom_ops), activation_calls=len(activation),
                activation_input_shapes=[dict(input_dims=json.loads(shape), calls=n) for shape,n in activation_shapes.items()],
                activation_input_types=[list(types) for types in sorted({tuple(e['args'].get('Input type', [])) for e in activation})],
                tail_cpu_ops=[dict(name=e['name'],input_types=e['args'].get('Input type'),input_dims=e['args'].get('Input Dims'))
                              for e in sorted(cpu,key=lambda e:e['ts'])[-30:]],
                activation_correlated_kernels=aggregate(activation_kernels), kernels=aggregate(kernels))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--step', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    main = analyze(args.main, args.step)
    candidate = analyze(args.candidate, args.step)
    result = dict(main=main, candidate=candidate,
                  standalone_gelu_calls_removed=main['gelu_cpu_calls']-candidate['gelu_cpu_calls'],
                  candidate_fusion_observed=candidate['fused_custom_op_calls'] > 0 and candidate['activation_calls'] > 0,
                  caveat='Diagnostic profile only. CPU and GPU use their corresponding complete step annotations; no whole-trace or E2E speedup inference.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('main','candidate')}, indent=2))
    for label, row in [('main',main),('candidate',candidate)]:
        print(json.dumps(dict(arm=label, **{k:row[k] for k in ['step','kernel_count','kernel_sum_ms','gelu_cpu_calls','fused_custom_op_calls','activation_calls']})))
