"""Keep synchronized diagnostics separate from profiler-free primary serving results."""
import json
import math
import statistics as stats
from pathlib import Path

R = Path(__file__).resolve().parents[1]
E = R / 'evidence'
def read(name):
    return json.loads((E / 'preflight' / name).read_text())
def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values)-1)*fraction
    lo = math.floor(pos)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(pos-lo)

primary = read('anima-rope-final-compare1-audit.json')
diagnostic = read('anima-rope-final-diagnostics1-audit.json')
assert primary['status'] == diagnostic['status'] == 'raw_contract_pass'
rows = []
for config in ['omni', 'main_eager', 'final_eager', 'main_bcg', 'final_bcg']:
    for prompt in ['short', 'long']:
        blocks = [x for x in primary['blocks'] if x['config']==config and x['prompt']==prompt]
        assert len(blocks)==2 and all(not x['denoise_is_synchronized'] for x in blocks)
        seconds = [v for b in blocks for v in b['http_seconds']]
        assert len(seconds)==8
        diag = [x for x in diagnostic['blocks'] if x['config']==config and x['prompt']==prompt]
        assert all(x['denoise_is_synchronized'] for x in diag)
        denoise = [v for b in diag for v in b['denoise_host_dispatch_seconds']]
        rows.append(dict(config=config, prompt=prompt, n=8, fresh_processes=2,
                         p50_seconds=stats.median(seconds), p95_seconds=percentile(seconds,.95),
                         requests_per_second=len(seconds)/sum(seconds),
                         peak_gpu_memory_gib=max(b['peak_gpu_memory_bytes'] for b in blocks)/2**30,
                         ready_seconds_range=[min(b['launch_to_ready_seconds'] for b in blocks),max(b['launch_to_ready_seconds'] for b in blocks)],
                         synchronized_denoise_median_seconds=stats.median(denoise) if denoise else None,
                         denoise_n=len(denoise)))
result=dict(rows=rows, paired=primary['paired'],
            method='Primary HTTP: SYNC0, no profiler, batch/concurrency1, full PNG/base64 response transfer included; decoding and disk writes excluded. Denoise: separate SYNC1 diagnostics, n4/config/prompt; not subtracted from primary E2E. Omni denoise not collected under the same synchronized SGLang stage observer.',
            readiness_caveat='Existing disk/compiler caches; no cold-start claim. Warmup and capture costs are separate.')
(E/'preflight/anima-rope-final-summary.json').write_text(json.dumps(result,indent=2)+'\n')
table=['| Configuration | Prompt | E2E p50 / p95 (s) | req/s | Peak GiB | Synchronized denoise (s) |',
       '|---|---|---:|---:|---:|---:|']
for x in rows:
    denoise = f"{x['synchronized_denoise_median_seconds']:.4f}" if x['denoise_n'] else 'not collected'
    table.append(f"| {x['config']} | {x['prompt']} | {x['p50_seconds']:.4f} / {x['p95_seconds']:.4f} | {x['requests_per_second']:.5f} | {x['peak_gpu_memory_gib']:.3f} | {denoise} |")
(E/'anima-rope-final-performance-table.md').write_text('\n'.join(table)+'\n')
print('\n'.join(table))
