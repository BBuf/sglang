"""Audit the frozen six-process native comparison without dropping measurements."""
import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path('/task')
plan_path = root/'sana-video-depthwise-v1-paired1.json'
plan = json.loads(plan_path.read_text())
status = json.loads((root/(plan['id']+'-status.json')).read_text())
assert status['status'] == 'raw_complete_audit_pending' and len(status['completed']) == 6
aa = json.loads((root/'evidence/preflight/sana-video-depthwise-main-eager-aa1-admission.json').read_text())
assert aa['passed']
blocks = []
for index, configuration in enumerate(plan['order']):
    name = f"{plan['id']}-{index:02d}-{configuration}"
    spec = plan['configurations'][configuration]
    metadata = json.loads((root/'block-runs'/name/'metadata.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['source_diff_sha256'] == spec.get('diff_sha256', hashlib.sha256(b'').hexdigest())
    for prompt in plan['measured_prompts']:
        directory = root/'outputs/benchmarks'/(name+'-'+prompt)
        rows = json.loads((directory/'results.json').read_text())
        meta = json.loads((directory/'metadata.json').read_text())
        assert meta['input_sha256'] == plan['sha256'][plan['contract']] and not meta['profiler']
        assert len(rows) == plan['warmups'] + plan['repeats']
        for row in rows:
            actual = hashlib.sha256((directory/f"{row['index']:02d}"/'output.mp4').read_bytes()).hexdigest()
            assert actual == row['sha256'] == spec['output_sha256'][prompt]
        samples = [row for row in rows if not row['warmup']]
        assert len(samples) == plan['repeats']
        values = [row['http_seconds'] for row in samples]
        blocks.append(dict(index=index, configuration=configuration, prompt=prompt, raw_seconds=values,
            mean_seconds=statistics.mean(values), median_seconds=statistics.median(values),
            peak_gpu_memory_bytes=max(row['peak_gpu_memory_bytes'] for row in samples)))

results = []
for prompt in plan['measured_prompts']:
    selected = [row for row in blocks if row['prompt'] == prompt]
    by_index = {row['index']: row for row in selected}
    means = {arm: statistics.mean([value for row in selected if row['configuration'] == arm for value in row['raw_seconds']])
             for arm in ['main_eager', 'candidate_eager', 'omni_eager']}
    paired = {}
    for comparison, pairs in plan['pair_indices'].items():
        ratios = [by_index[a]['median_seconds']/by_index[b]['median_seconds'] for a, b in pairs]
        logs = [math.log(ratio) for ratio in ratios]
        center = statistics.mean(logs)
        radius = 12.706204736432095*statistics.stdev(logs)/math.sqrt(2)
        paired[comparison] = dict(pairs=pairs, process_median_ratios=ratios,
            geometric_mean_ratio=math.exp(center), paired_log_t95_interval=[math.exp(center-radius), math.exp(center+radius)],
            interval_note='Two process pairs, df1; interval reflects between-pair uncertainty only.', all_pairs_faster=all(ratio > 1 for ratio in ratios))
    gain = 100*(1-means['candidate_eager']/means['main_eager'])
    results.append(dict(prompt=prompt, n_per_arm=6, mean_seconds=means,
        candidate_vs_main_latency_reduction_percent=gain,
        omni_over_candidate_speedup=means['omni_eager']/means['candidate_eager'],
        omni_over_main_speedup=means['omni_eager']/means['main_eager'], paired=paired,
        passes_pr_performance_gate=gain >= plan['admission']['candidate_mean_E2E_reduction_percent_min']
            and gain > aa['max_noise_percent'] and paired['main_vs_candidate']['all_pairs_faster']))
result = dict(passed=all(row['passes_pr_performance_gate'] for row in results),
    plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    aa_noise_percent=aa['max_noise_percent'], all_measured_MP4s_exact_qualified_native=True,
    all_raw_measurements_retained=True, rows=results, blocks=blocks,
    final_PR_admitted=False, note='V1 performance screening only. Cleanup/final retest, compatibility, complete public evidence and official Draft submission still required; Chinese numeric failure remains excluded.')
output = root/'evidence/preflight/sana-video-depthwise-v1-paired1-admission.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='blocks'}, indent=2))
