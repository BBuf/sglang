"""Audit every raw current-head request and admit only predeclared paired comparisons."""
import argparse
import base64
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'evidence'
parser = argparse.ArgumentParser()
parser.add_argument('size', type=int, choices=[512,1024])
parser.add_argument('phase', choices=['aa','primary'])
args = parser.parse_args()
run = f'qwen21-current-head-{args.size}-{args.phase}1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(values)-1)*fraction
    lo = int(position)
    return ordered[lo]+(ordered[min(lo+1,len(values)-1)]-ordered[lo])*(position-lo)


plan_path = EVIDENCE/'plans'/(run+'.json')
plan = read(plan_path)
receipt_path = EVIDENCE/'checkpoints'/(run+'-local-verification.json')
receipt = read(receipt_path)
assert receipt['all_member_hashes_verified'] and receipt['all_members_extracted']
reproduction = EVIDENCE/'reproduction'/run
assert digest(reproduction/(run+'.json')) == digest(plan_path)
status = read(reproduction/(run+'-status.json'))
assert status['status'] == 'raw_complete_audit_pending' and len(status['completed']) == len(plan['order'])
for path, expected in plan['sha256'].items():
    target = reproduction/Path(path).name if path.startswith('scripts/') else ROOT/path
    assert digest(target) == expected,path
for path in plan['required_admissions']:
    assert read(ROOT/path)['passed'],path
qualified = read(EVIDENCE/'qualification'/(run+'-raw-audit.json'))
assert qualified['status'] == 'raw_contract_pass' and qualified['source_and_native_backend_pass']
assert qualified['raw_output_count'] == sum(len(plan['configurations'][config].get('tasks',plan['tasks']))*4 for config in plan['order'])
uuids = ['GPU-1565b45e-a702-2015-9776-61f68c36fd59','GPU-7ca33735-71c8-fc89-57a6-c7cdbf8d28cd']
blocks, output_count = {task:[] for task in plan['tasks']}, 0
for i, config in enumerate(plan['order']):
    name = f'{run}-{i:02d}-{config}'
    spec = plan['configurations'][config]
    arm = spec['arm']
    block_folder = EVIDENCE/'block-runs'/name
    bm = read(block_folder/'metadata.json')
    assert bm['plan'] == plan and bm['index'] == i and bm['config'] == config and bm['status'] == 'complete'
    assert bm['source_sha'] == spec['head']
    assert bm['source_diff_sha256'] == digest(block_folder/'source.patch') == hashlib.sha256(b'').hexdigest()
    assert bm['runner_sha256'] == plan['sha256']['scripts/run_qwen21_current_head_blocks1.py']
    assert digest(block_folder/'launch.sh') == plan['sha256']['scripts/'+spec['launcher']]
    if arm == 'sgl':
        assert 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0' in (block_folder/'launch.sh').read_text()
    for task in spec.get('tasks',plan['tasks']):
        contract_path = ROOT/'inputs'/plan['contracts'][task]
        contract = read(contract_path)
        folder = EVIDENCE/'benchmarks'/(name+'-'+task)
        meta, rows, summary = (read(folder/(suffix+'.json')) for suffix in ('metadata','results','summary'))
        assert meta['source_sha'] == spec['head'] and meta['source_diff_sha256'] == bm['source_diff_sha256']
        assert meta['arm'] == arm and meta['physical_gpus'] == [0] and not meta['profiler']
        assert meta['harness_sha256'] == plan['sha256']['scripts/bench_qwen21_serving.py']
        assert meta['input_sha256'] == digest(contract_path) == plan['contract_sha256'][task]
        assert meta['args']['quality'] == plan['quality'] and meta['args']['repeats'] == plan['repeats'] and meta['args']['warmups'] == plan['warmups']
        assert len(rows) == plan['warmups']+plan['repeats']
        assert [row['index'] for row in rows] == list(range(len(rows)))
        allowed = set(meta['args']['allowed_gpu_pids'])
        assert allowed and allowed == set(bm['gpu_container_pids'][0]) and not bm['gpu_container_pids'][1]
        qualification = EVIDENCE/'qualification'/f'{arm}-qwen21-base-{task}-{name}-{plan["quality"]}'
        quality_rows = read(qualification/'results.json')
        short_hash = quality_rows[0]['sha256']
        assert digest(qualification/Path(quality_rows[0]['request']).name/'output.png') == short_hash
        values, memory, warmups = [], [], []
        for index,row in enumerate(rows):
            request_folder = folder/f'{index:02d}'
            request, response = read(request_folder/'request.json'), read(request_folder/'response.json')
            png = (request_folder/'output.png').read_bytes()
            assert hashlib.sha256(png).hexdigest() == row['sha256'] == short_hash
            assert base64.b64decode(response['data'][0]['b64_json'],validate=True) == png
            with Image.open(request_folder/'output.png') as image:
                assert image.mode == 'RGBA' and image.size == (contract['width'],contract['height'])
            for key in ('seed','num_inference_steps'):
                assert int(request[key]) == contract[key]
            assert request['generator_device'] == contract['generator_device'] == 'cpu'
            assert float(request['true_cfg_scale']) == contract['true_cfg_scale'] == 1
            assert request['negative_prompt'] == contract['negative_prompt']
            assert request['prompt'] == contract['prompts'][0]['prompt']
            assert request['model'] == contract['model_path'] and int(request['n']) == 1
            assert request['size'] == f'{contract["width"]}x{contract["height"]}'
            assert int(request['output_compression']) == (0 if arm == 'sgl' else 100)
            if arm == 'sgl':
                assert request['quality'] == plan['quality'] and request['profile'] in [False,'false']
            assert row['warmup'] == (index < plan['warmups'])
            samples = read(request_folder/'gpu-samples.json')
            assert len(samples) >= 2
            for sample in samples:
                g0,g1 = sample['gpus']
                assert [g0['uuid'],g1['uuid']] == uuids
                assert set(g0['pids']) <= allowed and not g1['pids']
            output_count += 1
            if row['warmup']:
                warmups.append(row['http_seconds'])
            else:
                values.append(row['http_seconds'])
                memory.append(max(sample['gpus'][0]['memory_bytes'] for sample in samples))
        assert summary['p50_seconds'] == statistics.median(values)
        blocks[task].append(dict(index=i,config=config,values=values,median=statistics.median(values),
            cv_percent=100*statistics.stdev(values)/statistics.mean(values),peak_gpu_memory_bytes=max(memory),
            launch_to_ready_seconds=bm['launch_to_ready_seconds'],warmup_http_seconds=warmups,
            qualification_http_seconds=[row['http_seconds'] for row in quality_rows]))
cases = {}
for task, rows in blocks.items():
    if args.phase == 'aa':
        assert len(rows) == 2 and all(row['config'] == 'main' for row in rows)
        noise = 100*(max(row['median'] for row in rows)/min(row['median'] for row in rows)-1)
        cv = max(row['cv_percent'] for row in rows)
        passed = noise <= 100*plan['aa_acceptance']['max_process_median_relative_difference'] and cv <= plan['aa_acceptance']['max_within_process_cv_percent']
        cases[task] = dict(passed=passed,max_noise_percent=noise,max_within_process_cv_percent=cv)
        continue
    aa = read(EVIDENCE/'preflight'/f'qwen21-current-head-{args.size}-aa1-admission.json')['cases'][task]
    assert aa['passed'] and len(rows) == 12
    arms, pairs = {}, {}
    for arm,spec in plan['configurations'].items():
        if task not in spec.get('tasks',plan['tasks']):
            continue
        selected = [row for row in rows if row['config'] == arm]
        values = [value for row in selected for value in row['values']]
        assert len(values) == plan['n_measured_per_arm'][arm]
        arms[arm] = dict(n=len(values),mean_seconds=statistics.mean(values),p50_seconds=statistics.median(values),
            p95_seconds=percentile(values,.95),serial_response_rate_per_second=1/statistics.mean(values),
            peak_gpu_memory_bytes=max(row['peak_gpu_memory_bytes'] for row in selected),
            max_cv_percent=max(row['cv_percent'] for row in selected))
    by_index = {row['index']:row for row in rows}
    for key,indices in plan['pair_indices'][task].items():
        ratios = [by_index[left]['median']/by_index[right]['median'] for left,right in indices]
        logs = [math.log(value) for value in ratios]
        center = statistics.mean(logs)
        radius = {2:12.706204736432095,4:3.182446305284263}[len(logs)]*statistics.stdev(logs)/math.sqrt(len(logs))
        pairs[key] = dict(indices=indices,ratios=ratios,geometric_mean=math.exp(center),
            log_t95_interval=[math.exp(center-radius),math.exp(center+radius)],df=len(logs)-1)
    gain = 100*(1-arms['current']['mean_seconds']/arms['main']['mean_seconds'])
    cleanup = 100*(arms['current']['mean_seconds']/arms['original']['mean_seconds']-1)
    upper = 100*(1/pairs['original_over_current']['log_t95_interval'][0]-1)
    endpoint = abs(rows[-1]['median']/rows[0]['median']-1)
    sgl_cv = max(row['cv_percent'] for row in rows if not row['config'].startswith('omni_'))
    noise_pass = endpoint <= plan['aa_acceptance']['max_process_median_relative_difference'] and sgl_cv <= plan['aa_acceptance']['max_within_process_cv_percent']
    all_pairs_faster = all(value > 1 for key in ('main_over_current','omni_over_current') for value in pairs[key]['ratios'])
    primary_pass = noise_pass and all_pairs_faster and gain > aa['max_noise_percent'] and gain >= plan['admission']['minimum_main_latency_reduction_percent']
    cleanup_pass = cleanup <= aa['max_noise_percent'] and upper < plan['admission']['cleanup_t95_regression_upper_percent_max']
    omni = plan['omni_configuration'][task]
    cases[task] = dict(passed=primary_pass and cleanup_pass,primary_passed=primary_pass,cleanup_passed=cleanup_pass,
        AAmax_noise_percent=aa['max_noise_percent'],main_endpoint_relative_difference=endpoint,main_current_noise_passed=noise_pass,
        final_vs_main_latency_reduction_percent=gain,omni_configuration=omni,omni_over_current_speedup=arms[omni]['mean_seconds']/arms['current']['mean_seconds'],
        cleanup_mean_latency_change_percent=cleanup,cleanup_t95_regression_upper_percent=upper,arms=arms,pairs=pairs,
        omni_cv_screen_passed=arms[omni]['max_cv_percent'] <= plan['aa_acceptance']['max_within_process_cv_percent'])
result = dict(run=run,source_sha=plan['current_head'],passed=all(row['passed'] for row in cases.values()),
    cases=cases,blocks=blocks,raw_benchmark_outputs_rehashed=output_count,raw_qualification_outputs=qualified['raw_output_count'],
    source_and_ownership_audit_passed=True,all_samples_retained=True,primary_stage_times_valid=False,final_PR_admitted=False,
    plan_sha256=digest(plan_path),archive_verification_sha256=digest(receipt_path),
    note='Fresh SYNC0 primary only; original historical failed/uncertain cleanup results remain retained. 1024 cases retain their smaller gains and are not required to reach1.5%;512 cases are. No threshold changes, selective retries or sample dropping. Separate compatibility, profiles, media and existing-PR update remain required.')
out = EVIDENCE/'preflight'/(run+'-admission.json')
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'blocks'},indent=2))
