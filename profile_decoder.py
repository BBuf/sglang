"""Attribute width-4096 decoder RMSNorm kernels to CPU operators on the native trace."""
from collections import Counter, defaultdict
import statistics

def summarize(events, head, count):
    stage, = [v for v in events if v.get('cat')=='user_annotation' and v.get('name')=='STAGE decoding_stage']
    op_name='sglang::rmsnorm_cast_before_weight_bitexact' if head else 'sglang::rmsnorm_mean_square_bitexact'
    calls=[v for v in events if v.get('cat')=='cpu_op' and v['name']==op_name
           and v.get('args',{}).get('Input Dims',[[]])[0]==[1,35200,4096]
           and stage['ts']<=v['ts']<stage['ts']+stage['dur']]
    ids={v['args']['External id'] for v in calls}
    assert len(ids)==len(calls)==count,(op_name,len(ids),len(calls),count)
    streams=defaultdict(list)
    for v in events:
        if v.get('cat')=='kernel':
            streams[(v['pid'],v['args'].get('stream'))].append(v)
    measurement=[]
    for dispatches in streams.values():
        dispatches.sort(key=lambda v:v['ts'])
        for i,kernel in enumerate(dispatches):
            if kernel['args'].get('External id') not in ids:continue
            assert kernel['name']==('_ordered_norm_kernel' if head else '_mean_square_kernel'),kernel['name']
            group=[kernel]
            if not head:
                scale=dispatches[i+1]
                assert 'to_copy_mul_rsqrt' in scale['name'] and scale['ts']>=kernel['ts']+kernel['dur']-0.5,scale
                group.append(scale)
            measurement.append(group)
    assert len(measurement)==count and len({v[0]['args']['External id'] for v in measurement})==count
    times=[sum(v['dur'] for v in row) for row in measurement]
    return dict(host_op=op_name,input=[1,35200,4096],occurrences=count,
                kernel_counts=dict(Counter(v['name'] for row in measurement for v in row)),
                total_gpu_ms=sum(times)/1000,median_gpu_us=statistics.median(times),
                stage_host_ms=stage['dur']/1000,
                method='Map the exact width-4096 CPU op to its device kernel using the profiler external ID. For the two-kernel main path include its immediate compiled scale on the same stream. Asynchronous device work after the host stage is included.')

if __name__ == '__main__':
    import argparse
    import gzip
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision', choices=['main', 'head'])
    parser.add_argument('trace', type=Path)
    options = parser.parse_args()
    reference = json.loads(Path(__file__).with_name('operator-and-rank-controls.json').read_text())
    count = reference['guards']['trace'][options.revision]['decoder_token_norm']['occurrences']
    with gzip.open(options.trace, 'rt') as file:
        events = json.load(file)['traceEvents']
    print(json.dumps(summarize(events, options.revision == 'head', count), indent=2))
