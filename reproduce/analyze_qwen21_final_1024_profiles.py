"""Match final-head complete GPU steps and verify packed MLP eager/BCG dispatch."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

from analyze_joy_gelu_profiles import aggregate, inside

ROOT = Path(__file__).resolve().parents[1]
RUN = 'qwen21-gate-up-v4-final-1024-diagnostics1'
HEAD = 'efcb88a201a41b1fecc516a60df0ca0a34fb6dd5'
rows = {}
for case in ['t2i1024','edit1024']:
    tokens = 1024 if case.endswith('512') else 4096
    arms = {}
    for index,config in enumerate(['main','final']):
        name = RUN+f'-{index:02d}-'+config
        source = json.loads((ROOT/'evidence/block-runs'/name/'metadata.json').read_text())
        assert source['status']=='complete'
        assert source['source_sha']==('af21e2b989e482cfdce39689126562f2209d53ac' if config=='main' else HEAD)
        folder = ROOT/'evidence/qualification'/f'sgl-qwen21-base-{case[:-4]}-{name}-profile-high'
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
    main,final = [arms[key] for key in ['main','final']]
    separate = sum(row['calls'] for row in main['mm_shapes'] if row['input_dims']==[[4096,4096],[4096,12288]])
    packed = sum(row['calls'] for row in final['mm_shapes'] if row['input_dims']==[[4096,4096],[4096,24576]])
    final_silu = sum(row['calls'] for row in final['kernels'] if '_packed_silu_mul_kernel' in row['name'])
    assert separate==64 and packed==32 and final_silu==32
    assert main['kernel_count']==416 and final['kernel_count']==384
    assert main['graph_launches']==final['graph_launches']==0
    assert main['output_sha256']==final['output_sha256']
    rows[case] = dict(arms=arms,dispatch_pass=True,baseline_gate_up_gemms=separate,final_packed_gemms=packed,final_packed_silu_calls=final_silu,native_qkv_preserved=True)
    print(case, 'GPUstep_ms', {k:v['gpu_step_ms'] for k,v in arms.items()}, 'kernels', {k:v['kernel_count'] for k,v in arms.items()})
dest = ROOT/'evidence/profiles/qwen21-gate-up-v4-final-1024-representative-profile-audit.json'
assert not dest.exists()
dest.write_text(json.dumps(dict(head=HEAD, all_dispatch_pass=True, cases=rows,
                                caveat='Fresh1024-only warmup preserves native packed QKV. Separate Torch profiles, excluded from primary E2E and separately timed native total denoise. No observer during these SGL profiles; no BCG performance claim.'),indent=2)+'\n')
