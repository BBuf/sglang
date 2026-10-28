"""Match final-head complete GPU steps and verify packed MLP eager/BCG dispatch."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

from analyze_joy_gelu_profiles import aggregate, inside

ROOT = Path(__file__).resolve().parents[1]
RUN = 'qwen21-gate-up-v4-compatibility1'
HEAD = 'efcb88a201a41b1fecc516a60df0ca0a34fb6dd5'
rows = {}
for case in ['t2i512','edit512','t2i1024','edit1024']:
    tokens = 1024 if case.endswith('512') else 4096
    arms = {}
    for config in ['main_eager','final_eager','final_bcg']:
        name = RUN+'-'+config
        source = json.loads((ROOT/'evidence/block-runs'/name/'metadata.json').read_text())
        assert source['status']=='complete'
        assert source['source_sha']==('af21e2b989e482cfdce39689126562f2209d53ac' if config=='main_eager' else HEAD)
        folder = ROOT/'evidence/qualification'/f'sgl-qwen21-base-{case}-{RUN}-{config}-profile-high'
        result = json.loads((folder/'results.json').read_text())[0]
        directory = folder/Path(result['request']).name
        assert hashlib.sha256((directory/'output.png').read_bytes()).hexdigest()==result['sha256']
        perf = json.loads((directory/'perf.json').read_text())
        paths = list((ROOT/'evidence/profiles'/name).glob(perf['request_id']+'*.gz'))
        assert len(paths)==1
        path = paths[0]
        raw = path.read_bytes()
        events = json.loads(gzip.decompress(raw))['traceEvents']
        cpu_anchors = [e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='ProfilerStep#3']
        gpu_anchors = [e for e in events if e.get('cat')=='gpu_user_annotation' and e.get('name')=='ProfilerStep#3']
        assert len(cpu_anchors)==len(gpu_anchors)==1
        ca,ga = cpu_anchors[0],gpu_anchors[0]
        assert ca['args']['External id']==ga['args']['External id']
        cpu = [e for e in events if e.get('cat')=='cpu_op' and inside(e,ca)]
        gpu = [e for e in events if e.get('cat')=='kernel' and inside(e,ga)]
        shapes = collections.Counter(json.dumps(e.get('args',{}).get('Input Dims')) for e in cpu if e['name']=='aten::mm')
        arms[config] = dict(source_sha=source['source_sha'], trace=str(path.relative_to(ROOT)),
                            trace_sha256=hashlib.sha256(raw).hexdigest(), output_sha256=result['sha256'],
                            cpu_step_ms=ca['dur']/1000,gpu_step_ms=ga['dur']/1000,
                            kernel_count=len(gpu),kernel_sum_ms=sum(e['dur'] for e in gpu)/1000,
                            kernels=aggregate(gpu),mm_shapes=[dict(input_dims=json.loads(s),calls=n) for s,n in shapes.items()],
                            graph_launches=sum(e.get('name')=='cudaGraphLaunch' and inside(e,ca) for e in events))
    main,final,bcg = [arms[key] for key in ['main_eager','final_eager','final_bcg']]
    separate = sum(row['calls'] for row in main['mm_shapes'] if row['input_dims']==[[tokens,4096],[4096,12288]])
    packed = sum(row['calls'] for row in final['mm_shapes'] if row['input_dims']==[[tokens,4096],[4096,24576]])
    final_silu = sum(row['calls'] for row in final['kernels'] if '_packed_silu_mul_kernel' in row['name'])
    bcg_silu = sum(row['calls'] for row in bcg['kernels'] if '_packed_silu_mul_kernel' in row['name'])
    assert separate==64 and packed==32 and final_silu==bcg_silu==32
    assert main['kernel_count']-final['kernel_count']==32
    assert main['graph_launches']==final['graph_launches']==0
    assert bcg['graph_launches']==(33 if case.startswith('t2i') else 0)
    assert main['output_sha256']==final['output_sha256']==bcg['output_sha256']
    rows[case] = dict(arms=arms, dispatch_pass=True, baseline_gate_up_gemms=separate, final_packed_gemms=packed,
                      final_packed_silu_calls=final_silu, bcg_packed_silu_calls=bcg_silu)
    print(case, 'GPUstep_ms', {k:v['gpu_step_ms'] for k,v in arms.items()}, 'kernels', {k:v['kernel_count'] for k,v in arms.items()}, 'graphs', bcg['graph_launches'])
dest = ROOT/'evidence/profiles/qwen21-gate-up-v4-final-profile-audit.json'
assert not dest.exists()
dest.write_text(json.dumps(dict(head=HEAD, all_dispatch_pass=True, cases=rows,
                                caveat='Separate observation/profiler diagnostics, excluded from primary E2E and total denoise claims. Native BCG edit fallback retained; no BCG performance claim.'),indent=2)+'\n')
