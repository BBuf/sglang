"""Validate archived native Qwen qualification artifacts against their frozen plan."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('run')
args = p.parse_args()
plan = json.loads((ROOT/'evidence/plans'/f'{args.run}.json').read_text())
records, manifests, metrics = [], [], []
for index, config in enumerate(plan['order']):
    block = f'{args.run}-{index:02d}-{config}'
    spec = plan['configurations'][config]
    bm = json.loads((ROOT/'evidence/block-runs'/block/'metadata.json').read_text())
    assert bm['status'] == 'complete'
    assert bm['source_sha'] == spec['head']
    assert bm['source_diff_sha256'] == spec.get('diff_sha256', hashlib.sha256(b'').hexdigest())
    log = (ROOT/'evidence/block-runs'/block/'server.log').read_text(errors='replace')
    if spec['arm'] == 'sgl':
        assert all(marker not in log for marker in ['Falling back to diffusers backend', 'Using diffusers backend', 'Loaded diffusers pipeline'])
    for task in plan['tasks']:
        folder = ROOT/'evidence/qualification'/f'{spec["arm"]}-qwen21-base-{task}-{block}-{plan["quality"]}'
        contract_bytes = (folder/'contract.json').read_bytes()
        contract = json.loads(contract_bytes)
        assert hashlib.sha256(contract_bytes).hexdigest() == plan['contract_sha256'][task]
        qm = json.loads((folder/'metadata.json').read_text())
        assert qm['source_sha'] == bm['source_sha'] and qm['source_diff_sha256'] == bm['source_diff_sha256']
        rows = json.loads((folder/'results.json').read_text())
        assert len(rows) == 4 and rows[0]['sha256'] == rows[-1]['sha256']
        prompts = contract['prompts'] + [contract['prompts'][0]]
        for row, prompt in zip(rows, prompts):
            directory = folder/Path(row['request']).name
            request = json.loads((directory/'request.json').read_text())['payload']
            assert request['prompt'] == prompt['prompt']
            assert int(request['seed']) == 1101 and int(request['num_inference_steps']) == 50
            assert int(request['n']) == 1 and float(request['true_cfg_scale']) == 1
            assert request['size'] == f'{contract["width"]}x{contract["height"]}'
            assert request['generator_device'] == 'cpu'
            assert int(request['output_compression']) == (0 if spec['arm']=='sgl' else 100)
            raw = (directory/'output.png').read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row['sha256']
            assert base64.b64decode(json.loads((directory/'response.json').read_text())['data'][0]['b64_json'], validate=True) == raw
            with Image.open(directory/'output.png') as image:
                assert image.mode == 'RGBA' and image.size == (contract['width'], contract['height'])
            records.append(dict(config=config, task=task, prompt=prompt['id'], sha256=row['sha256']))
        quality = folder.with_name(folder.name+'-quality.json')
        if quality.exists():
            values = json.loads(quality.read_text())
            assert values['all_pass']
            metrics.append(dict(config=config, task=task, results=values['results']))
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                manifests.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
out = ROOT/'evidence/qualification'/f'{args.run}-raw-audit.json'
assert not out.exists()
out.write_text(json.dumps(dict(status='raw_contract_pass', raw_output_count=len(records), source_and_native_backend_pass=True, native_RGBA_preserved=True, repeats_byte_identical=True, visual_review_pending=True, records=records, metrics=metrics, manifest=manifests), indent=2)+'\n')
print(json.dumps(dict(raw_output_count=len(records), quality_groups=[dict(config=m['config'], task=m['task'], min_rgb_ssim=min(x['rgb']['ssim'] for x in m['results']), min_rgb_psnr_db=min((x['rgb']['psnr_db'] for x in m['results'] if x['rgb']['psnr_db'] is not None), default=None), all_pixel_exact=all(x['pixel_exact'] for x in m['results'])) for m in metrics])))
