"""Retain both fixed cleanup experiments and all negative comparisons."""
import hashlib
import json
import math
import statistics as stats
from pathlib import Path

R=Path(__file__).resolve().parents[1]
E=R/'evidence'
def read(name):return json.loads((E/'preflight'/name).read_text())
a=read('anima-rope-final-compare1-audit.json')
b=read('anima-rope-cleanup-repeat1-audit.json')
aa=read('anima-main-cudnn-aa1-audit.json')
assert aa['all_aa_noise_pass']
assert a['status']==b['status']=='raw_contract_pass'
kernel='python/sglang/kernels/ops/diffusion/rope/rope_rotate_half_fp32.py'
before=(R/'candidates/anima-rope-rounding'/kernel).read_bytes()
after=(R/'candidates/anima-rope-rounding-final'/kernel).read_bytes()
assert before==after
rows=[]
for prompt in ['short','long']:
    original=a['paired'][prompt]['v1_eager_over_final_eager_cleanup']
    extra=b['paired'][prompt]['v1_over_final_cleanup']
    ratios=original['ratios']+extra['ratios']
    logs=[math.log(x) for x in ratios]
    mean=stats.mean(logs)
    # Four independent paired process medians, not 48 independent processes.
    half=3.182446305284263*stats.stdev(logs)/math.sqrt(len(logs))
    interval=[math.exp(mean-half),math.exp(mean+half)]
    gain=(math.exp(-mean)-1)*100
    rows.append(dict(prompt=prompt, original=original, supplementary=extra,
                     all_four_process_pairs=ratios, paired_geometric_mean=math.exp(mean),
                     final_latency_change_percent=gain,
                     descriptive_process_paired_t_interval_95=interval,
                     interval_contains_no_change=interval[0]<=1<=interval[1],
                     n_per_arm_per_prompt=24, fresh_processes_per_arm=4,
                     predeclared_one_percent_noninferiority_pass=interval[0]>1/1.01))
    assert interval[0]>1/1.01 and interval[0]<=1<=interval[1]
    for mode in ['eager','bcg']:
        value=a['paired'][prompt][f'main_{mode}_over_final_{mode}']
        assert value['all_pairs_faster'] and value['latency_reduction_percent']>10
result=dict(status='accepted_with_retained_negative_cleanup_point_estimates',
            rows=rows, executable_kernel_byte_identical=True,
            kernel_sha256=hashlib.sha256(after).hexdigest(),
            uncertainty_note='Both experiments retained, four pairs equally weighted. All cleanup pair point estimates are slightly negative. Descriptive log-ratio t intervals assume independent process pairs and include no change; only four pairs limits precision. No claim of exactly zero difference. Fixed supplementary ABBA was specified before running; no further favorable-result retries.',
            production_cleanup='Final commit adds operator documentation to unchanged V1 executable code; existing numerical helper reused and no gates or fallback policy altered.',
            final_model_gain_remains=True, new_bcg_net_win=False,
            omni_is_faster=True, chinese_cross_framework_quality_failed=True)
(E/'preflight/anima-rope-final-cleanup-admission.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
