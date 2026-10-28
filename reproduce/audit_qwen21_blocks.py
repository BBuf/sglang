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
    for task in plan['tasks']:
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
    vals = [[r['http_seconds'] for r in rows if r['task'] == task and r['block_index'] == i and not r['warmup']] for i in range(len(plan['order']))]
    med = [statistics.median(v) for v in vals]
    ratios = [med[a]/med[b] for a,b in plan.get('analysis_pair_indices', [[0,1],[3,2]] if len(med)==4 else [[0,1]])]
    checks[task] = dict(medians=med, cv_percent=[statistics.stdev(v)/statistics.mean(v)*100 for v in vals], paired_ratios=ratios, geometric_mean_ratio=statistics.geometric_mean(ratios), latency_reduction_percent=(1-1/statistics.geometric_mean(ratios))*100)
out = ROOT/'evidence/preflight'/f'{args.run}-audit.json'
assert not out.exists()
out.write_text(json.dumps(dict(status='raw_contract_pass', raw_output_count=len(rows), primary_output_count=sum(not r['warmup'] for r in rows), all_sgl_outputs_match_main_short_byte_exact=True, source_and_gpu_ownership_pass=True, checks=checks, rows=rows, manifest=manifest, caveat='HTTP primary only; asynchronous stage counters are not synchronized denoise. Admission requires separate noise/quality/profile/compatibility/cleanup review.'), indent=2)+'\n')
print(json.dumps(checks))
