"""Audit every frozen final measurement, including four cleanup process pairs."""
import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path('/task')
plan_path = root/'sana-video-depthwise-final-primary-cleanup1.json'
plan = json.loads(plan_path.read_text())
status = json.loads((root/(plan['id']+'-status.json')).read_text())
assert status['status']=='raw_complete_audit_pending' and len(status['completed'])==12
for name in plan['required_admissions']:
    assert json.loads((root/name).read_text())['passed'], name
aa = json.loads((root/'evidence/preflight/sana-video-depthwise-main-eager-aa1-admission.json').read_text())
blocks = []


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    low = int(position)
    high = min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(position-low)


for index, configuration in enumerate(plan['order']):
    name = f'{plan["id"]}-{index:02d}-{configuration}'
    spec = plan['configurations'][configuration]
    folder = root/'block-runs'/name
    metadata = json.loads((folder/'metadata.json').read_text())
    expected_head = spec.get('head','af21e2b989e482cfdce39689126562f2209d53ac' if spec['arm']=='sgl' else '817d4c17feb702289a2d2b2cbacfd594c262b7a7')
    assert metadata['status']=='complete' and metadata['source_sha']==expected_head
    assert metadata['source_diff_sha256']==spec.get('diff_sha256',hashlib.sha256(b'').hexdigest())
    for prompt in plan['measured_prompts']:
        directory = root/'outputs/benchmarks'/(name+'-'+prompt)
        rows = json.loads((directory/'results.json').read_text())
        meta = json.loads((directory/'metadata.json').read_text())
        assert meta['input_sha256']==plan['sha256'][plan['contract']] and not meta['profiler']
        assert meta['source_sha']==expected_head and meta['source_diff_sha256']==metadata['source_diff_sha256']
        assert len(rows)==plan['warmups']+plan['repeats']
        for row in rows:
            assert hashlib.sha256((directory/f'{row["index"]:02d}'/'output.mp4').read_bytes()).hexdigest()==row['sha256']==spec['output_sha256'][prompt]
        samples = [row for row in rows if not row['warmup']]
        assert len(samples)==plan['repeats']
        values = [row['http_seconds'] for row in samples]
        blocks.append(dict(index=index,configuration=configuration,prompt=prompt,raw_seconds=values,
                           mean_seconds=statistics.mean(values),median_seconds=statistics.median(values),
                           peak_gpu_memory_bytes=max(row['peak_gpu_memory_bytes'] for row in samples),
                           launch_to_ready_seconds=metadata['launch_to_ready_seconds']))
results = []
for prompt in plan['measured_prompts']:
    selected = [row for row in blocks if row['prompt']==prompt]
    by_index = {row['index']:row for row in selected}
    arm_summaries = {}
    for arm, n in plan['n_measured_per_arm_per_prompt'].items():
        values = [value for row in selected if row['configuration']==arm for value in row['raw_seconds']]
        assert len(values)==n
        mean = statistics.mean(values)
        arm_summaries[arm] = dict(n=n,mean_seconds=mean,p50_seconds=statistics.median(values),p95_seconds=percentile(values,.95),
                                 serial_response_rate_per_second=1/mean,cv_percent=100*statistics.stdev(values)/mean,
                                 peak_gpu_memory_bytes=max(row['peak_gpu_memory_bytes'] for row in selected if row['configuration']==arm))
    pairs_summary = {}
    for name, pairs in plan['pair_indices'].items():
        ratios = [by_index[left]['median_seconds']/by_index[right]['median_seconds'] for left,right in pairs]
        logs = [math.log(value) for value in ratios]
        t = {2:12.706204736432095,4:3.182446305284263}[len(pairs)]
        center = statistics.mean(logs)
        radius = t*statistics.stdev(logs)/math.sqrt(len(logs))
        pairs_summary[name] = dict(pairs=pairs,process_median_ratios=ratios,geometric_mean_ratio=math.exp(center),
                                   paired_log_t95_interval=[math.exp(center-radius),math.exp(center+radius)],
                                   df=len(pairs)-1,all_pairs_faster=all(value>1 for value in ratios),
                                   interval_note='Process-pair log-t interval; within-process requests are not treated as independent process replications.')
    means = {arm:row['mean_seconds'] for arm,row in arm_summaries.items()}
    gain = 100*(1-means['final_eager']/means['main_eager'])
    cleanup_change = 100*(means['final_eager']/means['v1_eager']-1)
    cleanup_upper = 100*(1/pairs_summary['v1_vs_final_cleanup']['paired_log_t95_interval'][0]-1)
    primary_pass = gain>=1.5 and gain>aa['max_noise_percent'] and pairs_summary['main_vs_final']['all_pairs_faster']
    cleanup_pass = cleanup_change<=aa['max_noise_percent'] and cleanup_upper<plan['admission']['cleanup_paired_t95_regression_upper_percent_max']
    results.append(dict(prompt=prompt,arms=arm_summaries,paired=pairs_summary,
                        final_vs_main_latency_reduction_percent=gain,omni_over_final_speedup=means['omni_eager']/means['final_eager'],
                        omni_over_main_speedup=means['omni_eager']/means['main_eager'],
                        cleanup_mean_latency_change_percent=cleanup_change,cleanup_paired_t95_latency_regression_upper_percent=cleanup_upper,
                        final_primary_passed=primary_pass,cleanup_passed=cleanup_pass))
result = dict(passed=all(row['final_primary_passed'] and row['cleanup_passed'] for row in results),
              rows=results,blocks=blocks,AAmax_noise_percent=aa['max_noise_percent'],all_saved_MP4s_exact_qualified_native=True,
              all_raw_measurements_and_warmups_retained=True,final_PR_admitted=False,
              plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              note='Final unprofiled serving evidence, not automatic PR admission. Denoise and capture costs come from separate diagnostics. No new BCGnetwin claim. Native Chinese crossOmni PSNRfailure remains in matrix. A failed or uncertain cleanup gate requires explicit supplemental evidence; no sample dropping or threshold change.')
out = root/'evidence/preflight/sana-video-depthwise-final-primary-cleanup1-admission.json'
assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='blocks'},indent=2))
