import json, math, statistics
from pathlib import Path
root=Path(__file__).parent
runs={x:json.loads((root/'results'/f'bench-{x}.json').read_text()) for x in ('A1','B1','B2','A2')}
rows={x:{r['name']:r for r in run['cases']} for x,run in runs.items()}
assert all(set(rows[x])==set(rows['A1']) for x in rows)
summary=[]
for name in rows['A1']:
    vals={x:rows[x][name] for x in rows}
    assert all(v['output']==vals['A1']['output'] for v in vals.values()), name
    a=statistics.mean(vals[x]['graph_us'] for x in ('A1','A2'))
    b=statistics.mean(vals[x]['graph_us'] for x in ('B1','B2'))
    ea=statistics.mean(vals[x]['eager_us'] for x in ('A1','A2'))
    eb=statistics.mean(vals[x]['eager_us'] for x in ('B1','B2'))
    summary.append(dict(name=name,baseline_graph_us=a,candidate_graph_us=b,graph_delta_pct=100*(b/a-1),baseline_eager_us=ea,candidate_eager_us=eb,eager_delta_pct=100*(eb/ea-1),output_equal=True,graph_runs={x:vals[x]['graph_us'] for x in vals}))
summary.sort(key=lambda r:r['graph_delta_pct'])
aggregate={'case_count':len(summary),'all_output_bytes_shapes_strides_equal':True,'graph_geomean_pct':100*(math.exp(statistics.mean(math.log(r['candidate_graph_us']/r['baseline_graph_us']) for r in summary))-1),'min_graph_delta_pct':summary[0]['graph_delta_pct'],'max_graph_delta_pct':summary[-1]['graph_delta_pct'],'cases':summary}
(root/'benchmark-summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
print(json.dumps({k:v for k,v in aggregate.items() if k!='cases'},indent=2))
for row in summary[-8:]: print(row['name'],round(row['graph_delta_pct'],2),row['graph_runs'])
