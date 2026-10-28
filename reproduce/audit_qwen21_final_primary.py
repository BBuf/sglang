"""Audit pinned paired runs and the saved request/response/GPU ownership contract."""
import argparse
import base64
import hashlib
import json
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('run')
args = p.parse_args()
plan = json.loads((ROOT / 'evidence/plans' / (args.run + '.json')).read_text())
rows, manifest = [], []
for i, config in enumerate(plan['order']):
    block = f'{args.run}-{i:02d}-{config}'
    spec = plan['configurations'][config]
    arm = spec['arm']
    bm = json.loads((ROOT/'evidence/block-runs'/block/'metadata.json').read_text())
    assert bm['status'] == 'complete'
    assert bm['source_sha'] == spec.get('head', 'af21e2b989e482cfdce39689126562f2209d53ac')
    assert bm['source_diff_sha256'] == spec.get('diff_sha256', hashlib.sha256(b'').hexdigest())
    for task in spec.get('tasks', plan['tasks']):
        d = ROOT/'evidence/benchmarks'/f'{block}-{task}'
        m = json.loads((d/'metadata.json').read_text())
        results = json.loads((d/'results.json').read_text())
        assert len(results) == plan['warmups'] + plan['repeats'] and not m['profiler']
        q = ROOT/'evidence/qualification'/f'{arm}-qwen21-base-{task}-{block}-{plan["quality"]}'
        qualified = json.loads((q/'results.json').read_text())[0]['sha256']
        main = ROOT/'evidence/qualification'/plan.get('main_qualification_references', {}).get(task, f'sgl-qwen21-base-{task}-mainhigh1')
        if arm == 'sgl':
            assert qualified == json.loads((main/'results.json').read_text())[0]['sha256']
        c = ROOT/'inputs'/plan.get('contracts', {}).get(task, f'qwen21-base-{task}-workloads.json')
        contract = json.loads(c.read_text())
        assert m['input_sha256'] == hashlib.sha256(c.read_bytes()).hexdigest() == plan['contract_sha256'][task]
        allowed = set(m['args']['allowed_gpu_pids'])
        assert allowed == set(bm['gpu_container_pids'][0])
        for index, row in enumerate(results):
            folder = d/f'{index:02d}'
            request = json.loads((folder/'request.json').read_text())
            resp = json.loads((folder/'response.json').read_text())
            png = (folder/'output.png').read_bytes()
            assert row['warmup'] == (index < plan['warmups'])
            assert hashlib.sha256(png).hexdigest() == row['sha256'] == qualified
            assert base64.b64decode(resp['data'][0]['b64_json'], validate=True) == png
            with Image.open(folder/'output.png') as im:
                assert im.mode == 'RGBA' and im.size == (contract['width'], contract['height'])
            for k, v in dict(seed=1101, num_inference_steps=50, n=1, output_compression=0 if arm=='sgl' else 100).items():
                assert int(request[k]) == v
            assert float(request['true_cfg_scale']) == 1
            assert request['generator_device'] == 'cpu'
            assert arm != 'sgl' or request['quality'] == plan['quality']
            assert request['prompt'] == contract['prompts'][0]['prompt']
            samples = json.loads((folder/'gpu-samples.json').read_text())
            assert samples
            for sample in samples:
                assert set(sample['gpus'][0]['pids']) <= allowed
                assert not sample['gpus'][1]['pids']
            rows.append(dict(block=block, block_index=i, config=config, task=task, index=index, warmup=row['warmup'], http_seconds=row['http_seconds'], sha256=row['sha256'], gpu_samples=len(samples)))
        for path in sorted(d.rglob('*')):
            if path.is_file():
                manifest.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
checks = {}
for task in plan['tasks']:
    blocks = []
    for i,config in enumerate(plan['order']):
        values = [r['http_seconds'] for r in rows if r['task']==task and r['block_index']==i and not r['warmup']]
        if not values: continue
        summary = json.loads((ROOT/'evidence/benchmarks'/f'{args.run}-{i:02d}-{config}-{task}'/'summary.json').read_text())
        blocks.append(dict(index=i,config=config,values=values,median=statistics.median(values),summary=summary))
    grouped = {arm:[b for b in blocks if b['config']==arm] for arm in ['main','final']}
    grouped['omni'] = [b for b in blocks if b['config'].startswith('omni_')]
    assert all(len(v)==2 for v in grouped.values())
    comparisons = {}
    for baseline in ['main','omni']:
        ratios = [a['median']/b['median'] for a,b in zip(grouped[baseline],grouped['final'])]
        mean = statistics.geometric_mean(ratios)
        comparisons[baseline+'_over_final'] = dict(paired_ratios=ratios,geometric_mean_ratio=mean,latency_reduction_percent=(1-1/mean)*100)
    endpoint_drift=abs(grouped['main'][1]['median']/grouped['main'][0]['median']-1)
    max_cv=max(b['summary']['cv_percent'] for b in blocks)
    limits=plan['aa_acceptance']
    noise_pass=endpoint_drift<=limits['max_process_median_relative_difference'] and max_cv<=limits['max_within_process_cv_percent']
    checks[task] = dict(blocks=blocks,comparisons=comparisons,main_endpoint_relative_difference=endpoint_drift,max_within_process_cv_percent=max_cv,noise_pass=noise_pass)
out = ROOT/'evidence/preflight'/f'{args.run}-audit.json'
assert not out.exists()
out.write_text(json.dumps(dict(status='raw_contract_pass', raw_output_count=len(rows), primary_output_count=sum(not r['warmup'] for r in rows), all_sgl_outputs_match_main_short_byte_exact=True, source_and_gpu_ownership_pass=True, checks=checks, rows=rows, manifest=manifest, caveat='HTTP primary only; asynchronous stage counters are not synchronized denoise. Admission requires separate noise/quality/profile/compatibility/cleanup review.'), indent=2)+'\n')
print(json.dumps(checks))
