import json, math, statistics, sys
from pathlib import Path
root=Path(sys.argv[1])
runs={arm:json.loads((root/f'cold-{arm}.json').read_text()) for arm in ('A1','B1','B2','A2')}
rows={arm:{c['name']:c for c in run['cases']} for arm,run in runs.items()}
assert all(set(x)==set(rows['A1']) for x in rows.values())
cases=[]
for name in rows['A1']:
    values={a:r[name] for a,r in rows.items()}
    equal=all(v['output']==values['A1']['output'] and v['replay_output_equal'] for v in values.values())
    assert equal,name
    times={a:v['cold_median_us'] for a,v in values.items()}
    before=statistics.mean([times['A1'],times['A2']]);after=statistics.mean([times['B1'],times['B2']])
    model=name.startswith(('qwen_model_','model_'))
    cases.append(dict(name=name,group='model_helper' if model else 'direct_kernel',baseline_us=before,candidate_us=after,delta_us=after-before,delta_pct=100*(after/before-1),baseline_drift_pct=100*(times['A2']/times['A1']-1),all_outputs_equal=equal,medians_us=times))
def aggregate(group):
    if not group: return None
    group=sorted(group,key=lambda r:r['delta_pct'])
    return dict(count=len(group),min_pct=group[0]['delta_pct'],max_pct=group[-1]['delta_pct'],geomean_pct=100*(math.exp(statistics.mean(math.log(r['candidate_us']/r['baseline_us']) for r in group))-1))
result=dict(case_count=len(cases),all_output_bytes_shapes_strides_equal=True,l2_bytes=runs['A1']['l2_bytes'],eviction_bytes=runs['A1']['eviction_bytes'],empty_event_interval_us={a:r['empty_interval']['cold_median_us'] for a,r in runs.items()},direct_kernels=aggregate([c for c in cases if c['group']=='direct_kernel']),model_helpers=aggregate([c for c in cases if c['group']=='model_helper']),cases=sorted(cases,key=lambda r:r['delta_pct']))
(root/(sys.argv[2] if len(sys.argv)>2 else 'cold-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
print('Largest increases:')
for c in result['cases'][-12:]:print(c['name'],round(c['delta_pct'],2),round(c['delta_us'],3),c['medians_us'])
