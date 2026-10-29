"""Audit all saved primary A/A requests and the predeclared noise gates."""
import hashlib
import json
import statistics
from pathlib import Path

root = Path('/task')
plan_path = root/'sana-video-depthwise-main-eager-aa1.json'
plan = json.loads(plan_path.read_text())
status = json.loads((root/(plan['id']+'-status.json')).read_text())
assert status['status'] == 'raw_complete_audit_pending'
assert len(status['completed']) == 2
rows = []
for index, configuration in enumerate(plan['order']):
    block = f"{plan['id']}-{index:02d}-{configuration}"
    spec = plan['configurations'][configuration]
    source = root/'block-runs'/block
    metadata = json.loads((source/'metadata.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['source_sha'] == 'af21e2b989e482cfdce39689126562f2209d53ac'
    assert metadata['source_diff_sha256'] == hashlib.sha256(b'').hexdigest()
    assert 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0' in (source/'launch.sh').read_text()
    for prompt in plan['measured_prompts']:
        folder = root/'outputs/benchmarks'/(block+'-'+prompt)
        results = json.loads((folder/'results.json').read_text())
        meta = json.loads((folder/'metadata.json').read_text())
        assert meta['input_sha256'] == plan['sha256'][plan['contract']]
        assert not meta['profiler']
        assert len(results) == plan['warmups'] + plan['repeats']
        for result in results:
            actual = hashlib.sha256((folder/f"{result['index']:02d}"/'output.mp4').read_bytes()).hexdigest()
            assert actual == result['sha256'] == spec['output_sha256'][prompt]
        values = [row['http_seconds'] for row in results if not row['warmup']]
        assert len(values) == plan['repeats'] and all(value > 0 for value in values)
        cv = 100*statistics.stdev(values)/statistics.mean(values)
        middle = len(values)//2
        drift = 100*abs(statistics.median(values[:middle])/statistics.median(values[middle:])-1)
        rows.append(dict(block=block, prompt=prompt, n=len(values), raw_seconds=values,
            mean_seconds=statistics.mean(values), median_seconds=statistics.median(values),
            cv_percent=cv, half_median_difference_percent=drift,
            passed=cv <= plan['noise_policy']['cv_percent_max'] and drift <= plan['noise_policy']['half_median_difference_percent_max']))
process_pairs = []
for prompt in plan['measured_prompts']:
    pair = [row for row in rows if row['prompt'] == prompt]
    assert len(pair) == 2
    drift = 100*abs(pair[0]['median_seconds']/pair[1]['median_seconds']-1)
    process_pairs.append(dict(prompt=prompt, process_median_difference_percent=drift,
        passed=drift <= plan['noise_policy']['between_process_median_difference_percent_max']))
result = dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    passed=all(row['passed'] for row in rows+process_pairs),
    rows=rows, process_pairs=process_pairs,
    max_noise_percent=max([row['cv_percent'] for row in rows]+[row['half_median_difference_percent'] for row in rows]+[row['process_median_difference_percent'] for row in process_pairs]),
    all_saved_MP4s_exact_native=True, candidate_speedup_claim=False,
    note='Fresh native-main A/A only; all timings unprofiled, GPU0,3measurements per English prompt per process,2processes. Raw warmups retained and excluded as declared.')
output = root/'evidence/preflight/sana-video-depthwise-main-eager-aa1-admission.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
