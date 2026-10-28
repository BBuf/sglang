"""Audit the fixed within-process cleanup experiment; retain every quartet."""
import argparse
import base64
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('size', type=int, choices=[512, 1024])
args = p.parse_args()
run = f'qwen21-cleanup-within-{args.size}-1'
plan = json.loads((ROOT/'evidence/plans'/f'{run}.json').read_text())
schedule = json.loads((ROOT/'evidence/plans'/plan['schedule']).read_text())
name = run+'-00-final'
source = json.loads((ROOT/'evidence/block-runs'/name/'metadata.json').read_text())
assert source['status']=='complete'
assert source['source_sha']==plan['configurations']['final']['head']
assert source['source_diff_sha256']==hashlib.sha256(b'').hexdigest()
manifest = json.loads((ROOT/'evidence/preflight/qwen21-cleanup-source-equivalence.json').read_text())
observed = [json.loads(line) for path in (ROOT/'evidence/diagnostics/qwen21-cleanup-within-process1').glob('worker-*.jsonl') for line in path.read_text().splitlines()]
by_id = {row['request_id']:row for row in observed if not row['is_warmup']}
checks = {}
rows = []
for task in plan['tasks']:
    folder = ROOT/'evidence/benchmarks'/f'{name}-{task}'
    metadata = json.loads((folder/'metadata.json').read_text())
    assert metadata['profiler'] and metadata['instrumented_cleanup_only']
    assert metadata['cleanup_schedule']==schedule
    assert metadata['input_sha256']==plan['contract_sha256'][task]
    contract = json.loads((ROOT/'inputs'/plan['contracts'][task]).read_text())
    reference = ROOT/'evidence/qualification'/plan['main_qualification_references'][task]
    expected_png = json.loads((reference/'results.json').read_text())[0]['sha256']
    results = json.loads((folder/'results.json').read_text())
    assert len(results)==len(schedule['order'])
    allowed = set(metadata['args']['allowed_gpu_pids'])
    assert allowed == set(source['gpu_container_pids'][0])
    for i,row in enumerate(results):
        directory = folder/f'{i:02d}'
        request = json.loads((directory/'request.json').read_text())
        response = json.loads((directory/'response.json').read_text())
        perf = json.loads((directory/'perf.json').read_text())
        record = by_id[perf['request_id']]
        assert record['variant']==row['variant']==schedule['order'][i]
        assert record['hook_sha256']==plan['observer_sha256']
        assert record['function_sha256']==manifest['functions']['v3' if row['variant']=='precleanup' else 'v4']
        assert record['width']==record['height']==args.size
        assert row['warmup']==(i<plan['warmups'])
        png = (directory/'output.png').read_bytes()
        assert hashlib.sha256(png).hexdigest()==row['sha256']==expected_png
        assert base64.b64decode(response['data'][0]['b64_json'],validate=True)==png
        with Image.open(directory/'output.png') as image:
            assert image.mode=='RGBA' and image.size==(args.size,args.size)
        for k,v in dict(seed=1101,num_inference_steps=50,n=1,output_compression=0).items():
            assert int(request[k])==v
        assert float(request['true_cfg_scale'])==1
        assert request['quality']=='high' and request['generator_device']=='cpu'
        assert request['prompt']==contract['prompts'][0]['prompt']
        for sample in json.loads((directory/'gpu-samples.json').read_text()):
            assert set(sample['gpus'][0]['pids']) <= allowed
            assert not sample['gpus'][1]['pids']
        rows.append(dict(task=task,request_id=perf['request_id'],observer=record,**row))
    measured = [row for row in results if not row['warmup']]
    quartets = []
    for start in range(0,len(measured),4):
        group = measured[start:start+4]
        assert [row['variant'] for row in group]==['precleanup','final','final','precleanup']
        values = [row['http_seconds'] for row in group]
        ratio = math.sqrt(values[1]*values[2]/(values[0]*values[3]))
        quartets.append(dict(index=start//4,http_seconds=values,after_over_before=ratio))
    logs = [math.log(q['after_over_before']) for q in quartets]
    mu = statistics.mean(logs)
    sem = statistics.stdev(logs)/math.sqrt(len(logs))
    critical = {8:2.364624251,6:2.570581836}[len(logs)]
    interval = [math.exp(mu-critical*sem), math.exp(mu+critical*sem)]
    checks[task] = dict(quartets=quartets,after_over_before=math.exp(mu),t95_interval=interval,
                        upper_bound_below_predeclared_1_001=interval[1]<=1.001)
result = dict(run=run,source_sha=source['source_sha'],raw_contract_pass=True,
              raw_outputs=len(rows),instrumented_cleanup_only=True,primary_timing=False,
              checks=checks,all_tasks_pass=all(c['upper_bound_below_predeclared_1_001'] for c in checks.values()),rows=rows,
              caveat='Causal within-process native-function experiment; previous independent-process screen failures remain published. This does not replace uninstrumented final main/PR/Omni measurements.')
out = ROOT/'evidence/preflight'/f'{run}-audit.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({task:{k:v for k,v in check.items() if k!='quartets'} for task,check in checks.items()}))
