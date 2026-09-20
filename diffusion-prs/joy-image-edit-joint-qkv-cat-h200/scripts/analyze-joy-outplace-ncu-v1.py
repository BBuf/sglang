"""Read paired NCU evidence without using profiler timings as model E2E."""
import json
from pathlib import Path
import sys
import time

root=Path('/campaign/profile/joy-outplace-v1-h200')
done=Path('/campaign/ncu-joy-outplace-v1.exit')
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
sys.path.insert(0,'/opt/nvidia/nsight-compute/2025.3.1/extras/python')
import ncu_report

keys=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum',
 'dram__throughput.avg.pct_of_peak_sustained_elapsed','sm__throughput.avg.pct_of_peak_sustained_elapsed',
 'lts__throughput.avg.pct_of_peak_sustained_elapsed','launch__registers_per_thread','launch__block_size',
 'launch__grid_size','launch__waves_per_multiprocessor','sm__warps_active.avg.pct_of_peak_sustained_active',
 'smsp__warps_eligible.avg.per_cycle_active','smsp__issue_active.avg.pct_of_peak_sustained_active',
 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
reports={}
for arm in ['baseline','candidate']:
 rep=ncu_report.load_report(str(root/'reports'/f'{arm}.ncu-rep'));rng=rep.range_by_idx(0);rows=[]
 for idx in range(rng.num_actions()):
  action=rng.action_by_idx(idx);all_metrics={}
  for name in action.metric_names():
   metric=action[name]
   try:all_metrics[name]=dict(value=metric.value(),unit=metric.unit())
   except Exception:continue
  (root/'analysis'/f'{arm}-{idx}-all-metrics.json').write_text(json.dumps(all_metrics,indent=1,default=str))
  selected=keys+[k for k in all_metrics if 'warps_issue_stalled' in k and k.endswith('.ratio')]
  rows.append(dict(name=action.name(),metrics={k:all_metrics[k] for k in selected if k in all_metrics}))
 assert len(rows)==(7 if arm=='baseline' else 3),(arm,len(rows))
 totals={k:sum(r['metrics'][k]['value'] for r in rows) for k in ['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum']}
 reports[arm]=dict(kernels=rows,totals=totals)
(root/'analysis/key-metrics.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2),flush=True)
