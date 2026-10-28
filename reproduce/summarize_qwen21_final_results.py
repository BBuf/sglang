"""Join separately audited primary, quality and native-stage evidence for the PR."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'
def read(path):return json.loads(path.read_text())
def quantile(values,q):
    values=sorted(values)
    position=(len(values)-1)*q
    lo=int(position)
    hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)

cases={}
for size in [512,1024]:
    run=f'qwen21-gate-up-v4-final-{size}-compare1'
    plan=read(E/'plans'/f'{run}.json')
    primary=read(E/'preflight'/f'{run}-audit.json')
    admission=read(E/'preflight'/f'{run}-admission.json')
    quality=read(E/'qualification'/f'{run}-raw-audit.json')
    diagnostics=read(E/'preflight'/f'qwen21-gate-up-v4-final-{size}-diagnostics1-audit.json')
    assert primary['status']=='raw_contract_pass' and diagnostics['all_raw_audits_pass']
    assert quality['status']=='raw_contract_pass'
    for task,check in primary['checks'].items():
        assert admission['cases'][task]['accepted']
        arms={}
        for arm in ['main','final','omni']:
            blocks=[b for b in check['blocks'] if (b['config'].startswith('omni_') if arm=='omni' else b['config']==arm)]
            assert len(blocks)==2
            values=[v for block in blocks for v in block['values']]
            startup=[]
            for block in blocks:
                name=f'{run}-{block["index"]:02d}-{block["config"]}'
                spec=plan['configurations'][block['config']]
                metadata=read(E/'block-runs'/name/'metadata.json')
                qualification=read(E/'qualification'/f'{spec["arm"]}-qwen21-base-{task}-{name}-high'/'results.json')
                measured=read(E/'benchmarks'/f'{name}-{task}'/'results.json')
                startup.append(dict(block=name,launch_to_ready_seconds=metadata['launch_to_ready_seconds'],qualification_seconds=[r['http_seconds'] for r in qualification],explicit_warmup_seconds=[r['http_seconds'] for r in measured if r['warmup']]))
            config=blocks[0]['config']
            denoise=diagnostics['checks'][config]['tasks'][task]
            arms[arm]=dict(config=config,source_sha=plan['configurations'][config]['head'],n=len(values),fresh_processes=2,primary_p50_seconds=statistics.median(values),primary_p95_seconds=quantile(values,.95),primary_mean_seconds=statistics.mean(values),sequential_requests_per_second=len(values)/sum(values),peak_sampled_nvml_mib=max(b['summary']['peak_gpu_memory_bytes'] for b in blocks)/(1024**2),separate_native_denoise_p50_seconds=denoise['denoise_p50_seconds'],separate_native_denoise_n=len(denoise['denoise_seconds']),startup_and_warmups=startup)
        metrics=[row for group in quality['metrics'] if group['task']==task and group['config'].startswith('omni_') for row in group['results']]
        assert metrics and all(row['pass_gate'] for row in metrics)
        cross={channel:{metric:min((row[channel][metric] for row in metrics if row[channel][metric] is not None),default=None) for metric in ['ssim','psnr_db']} for channel in ['rgb','alpha']}
        cases[f'{task}{size}']=dict(arms=arms,paired_comparisons=check['comparisons'],noise_admission=admission['cases'][task],main_endpoint_drift_percent=check['main_endpoint_relative_difference']*100,max_within_process_cv_percent=check['max_within_process_cv_percent'],cross_framework_minimum_metrics=cross,main_final_RGBA_byte_exact=primary['all_sgl_outputs_match_main_short_byte_exact'])
out=E/'preflight/qwen21-gate-up-v4-final-results-summary.json'
assert not out.exists()
out.write_text(json.dumps(dict(head='efcb88a201a41b1fecc516a60df0ca0a34fb6dd5',cases=cases,note='Primary HTTP statistics use uninstrumented requests; denoise comes from separate native synchronized diagnostic runs. Do not subtract these separately measured values or treat stage timing as primary E2E. p95 is descriptive for16/12 requests; paired ratios use per-process medians.'),indent=2)+'\n')
for case,row in cases.items():
    print(case,{arm:round(values['primary_p50_seconds'],6) for arm,values in row['arms'].items()},row['paired_comparisons'])
