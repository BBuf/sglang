"""Select each measured kernel explicitly; generic helpers default to action zero."""
import json
from pathlib import Path
import sys
import time

root = Path('/campaign/profile/joy-outplace-v1-h200')
done = Path('/campaign/ncu-joy-outplace-v1.exit')
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
sys.path.insert(0, '/opt/nvidia/nsight-compute/2025.3.1/extras/python')
import ncu_report
from extract_stall_hotspots import collect_per_pc, aggregate_by_source_line, write_report
from ncu_utils import per_instance_values
from plot_timeline import ascii_plot

source = ncu_report.load_report(str(root / 'reports/candidate-source.ncu-rep'))
source_range = source.range_by_idx(0)
assert source_range.num_actions() == 3
for index, tag in [(1, 'qk'), (2, 'cat')]:
    action = source_range.action_by_idx(index)
    assert ('qknorm' if tag == 'qk' else 'joint_qkv') in action.name()
    per_line = aggregate_by_source_line(action, collect_per_pc(action))
    write_report(per_line, root / f'analysis/stall_hotspots_{tag}.txt', tag, top_n=25)
    print(tag, action.name(), 'source lines', len(per_line), flush=True)

metrics = [
    'LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'FBSP.TriageCompute.dramc__throughput.avg.pct_of_peak_sustained_elapsed',
]
raw = {}
plot_lines = []
for arm, selected in [('baseline', [(3,'qk'),(6,'v_cat')]), ('candidate', [(1,'qk'),(2,'cat')])]:
    report = ncu_report.load_report(str(root / 'reports' / f'{arm}.ncu-rep'))
    rng = report.range_by_idx(0)
    for index, tag in selected:
        action = rng.action_by_idx(index)
        key = f'{arm}-{tag}'
        raw[key] = dict(action_index=index, kernel=action.name(), metrics={})
        plot_lines.append(f'\n{key}: action {index}, {action.name()}')
        for metric in metrics:
            values = per_instance_values(action, metric)
            raw[key]['metrics'][metric] = values
            if values is None:
                plot_lines.append(metric+': no instances')
            else:
                # cols=len(values) prevents the helper from dropping a remainder.
                plot_lines.extend(ascii_plot(values, metric, 15, len(values)))
                print(key, metric, 'instances', len(values), flush=True)
(root / 'analysis/pm-instances.json').write_text(json.dumps(raw, indent=2))
(root / 'analysis/pm_timeline_plots.txt').write_text('\n'.join(plot_lines))
