"""Extract every profiled post-op, key metrics, PM samples, and source stalls."""
import collections
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,'/opt/nvidia/nsight-compute/2025.3.1/extras/python')
import ncu_report

run=Path(__file__).resolve().parents[1]
keys=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed','sm__throughput.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed','launch__registers_per_thread','launch__block_size','launch__grid_size','launch__waves_per_multiprocessor','sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active','smsp__issue_active.avg.pct_of_peak_sustained_active','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']

def val(m,i):
    if m.kind()==m.ValueKind_UINT64:return m.as_uint64(i)
    if m.kind() in (m.ValueKind_DOUBLE,m.ValueKind_FLOAT):return m.as_double(i)
    return m.as_string(i)

all_reports={}
for provider in ('baseline','fused'):
    rep=ncu_report.load_report(str(run/f'reports/full-{provider}.ncu-rep'))
    rng=rep.range_by_idx(0);actions=[]
    for i in range(rng.num_actions()):
        a=rng.action_by_idx(i);metrics={};pm={}
        for name in a.metric_names():
            m=a[name]
            try:metrics[name]=dict(value=m.value(),unit=m.unit())
            except Exception:continue
            if name.startswith('pmsampling:'):
                count=m.num_instances()
                values=[val(m,j) for j in range(count)]
                numeric=[x for x in values if isinstance(x,(int,float))]
                pm[name]=dict(samples=count,values=values)
                if numeric:pm[name].update(min=min(numeric),max=max(numeric),mean=statistics.mean(numeric))
        (run/f'analysis/all-metrics-{provider}-{i}.json').write_text(json.dumps(metrics,indent=1,default=str))
        (run/f'analysis/pm-samples-{provider}-{i}.json').write_text(json.dumps(pm,indent=1,default=str))
        selected=keys+[n for n in metrics if 'warp_issue_stalled' in n and n.endswith('pct')]
        actions.append(dict(name=a.name(),metrics={n:metrics.get(n) for n in selected},pm_metrics=len(pm)))
    assert len(actions)==(5 if provider=='baseline' else 1),(provider,len(actions))
    totals={k:sum(a['metrics'][k]['value'] for a in actions) for k in ['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum']}
    all_reports[provider]=dict(actions=actions,totals=totals)

rep=ncu_report.load_report(str(run/'reports/source-fused.ncu-rep'));a=rep.range_by_idx(0).action_by_idx(0)
lines=collections.defaultdict(collections.Counter)
for name in a.metric_names():
    if not name.startswith('smsp__pcsamp_warps_issue_stalled_'):continue
    m=a[name]
    if not m.num_instances() or not m.has_correlation_ids():continue
    cor=m.correlation_ids()
    for i in range(m.num_instances()):
        pc=cor.as_uint64(i);v=val(m,i)
        if not isinstance(v,(int,float)) or not v:continue
        si=a.source_info(pc)
        loc=(si.file_name(),si.line()) if si is not None else ('unknown',0)
        lines[loc][name]+=v
hotspots=[dict(file=k[0],line=k[1],samples=sum(v.values()),stalls=dict(v)) for k,v in lines.items()]
hotspots.sort(key=lambda r:-r['samples'])
(run/'analysis/stall-hotspots-fused.json').write_text(json.dumps(hotspots,indent=2))
(run/'analysis/key-metrics.json').write_text(json.dumps(all_reports,indent=2))
print(json.dumps(all_reports,indent=2))
