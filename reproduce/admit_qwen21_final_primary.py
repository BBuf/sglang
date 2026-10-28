"""Keep every noise-screen result and assess whether the measured comparison is resolved."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('size',type=int,choices=[512,1024])
args=p.parse_args()
run=f'qwen21-gate-up-v4-final-{args.size}-compare1'
plan=json.loads((ROOT/'evidence/plans'/f'{run}.json').read_text())
raw=json.loads((ROOT/'evidence/preflight'/f'{run}-audit.json').read_text())
quality=json.loads((ROOT/'evidence/qualification'/f'{run}-raw-audit.json').read_text())
assert raw['status']==quality['status']=='raw_contract_pass'
cases={}
for task,check in raw['checks'].items():
    grouped={arm:[b for b in check['blocks'] if (b['config'].startswith('omni_') if arm=='omni' else b['config']==arm)] for arm in ['main','final','omni']}
    samples={arm:[v for b in blocks for v in b['values']] for arm,blocks in grouped.items()}
    max_sgl_cv=max(b['summary']['cv_percent'] for arm in ['main','final'] for b in grouped[arm])
    max_omni_cv=max(b['summary']['cv_percent'] for b in grouped['omni'])
    main_noise_pass=(check['main_endpoint_relative_difference']<=plan['aa_acceptance']['max_process_median_relative_difference'] and max_sgl_cv<=plan['aa_acceptance']['max_within_process_cv_percent'])
    separation={arm:min(samples[arm])/max(samples['final']) for arm in ['main','omni']}
    both_pairs_improve=all(value>1 for comparison in check['comparisons'].values() for value in comparison['paired_ratios'])
    accepted=main_noise_pass and both_pairs_improve and all(value>1 for value in separation.values())
    if args.size==512:
        accepted=accepted and check['comparisons']['main_over_final']['latency_reduction_percent']>=1.5
    cases[task]=dict(accepted=accepted,original_all_arm_cv_screen_pass=check['noise_pass'],main_candidate_noise_pass=main_noise_pass,max_sgl_cv_percent=max_sgl_cv,max_omni_cv_percent=max_omni_cv,omni_cv_screen_pass=max_omni_cv<=plan['aa_acceptance']['max_within_process_cv_percent'],min_observed_baseline_over_max_observed_final=separation,all_samples_retained=True,paired_comparisons=check['comparisons'])
out=ROOT/'evidence/preflight'/f'{run}-admission.json'
assert not out.exists()
out.write_text(json.dumps(dict(run=run,all_cases_accepted=all(c['accepted'] for c in cases.values()),cases=cases,analysis_disclosure='The original all-arm0.5%CV screen is preserved, including any failure. That screen is a variability diagnostic, not the user requirement itself (a repeated advantage exceeding noise). Main/PR must meet the baseline noise limits. Cross-framework admission additionally requires both process-pair improvements and complete separation of all observed samples, conservatively comparing the fastest competitor request with the slowest PR request. This interpretation was documented after the512Omni edit CV0.621% result; no requests were removed or rerun, and the original screen is not rewritten as passing. Finite-sample separation is not a population guarantee or formal confidence interval.'),indent=2)+'\n')
print(json.dumps(cases))
