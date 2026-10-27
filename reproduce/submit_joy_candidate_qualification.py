"""Frozen image-edit requests. Qualification timings are not benchmarks."""
import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=['sgl', 'omni'])
    p.add_argument('--label', required=True)
    p.add_argument('--contract', type=Path, default=Path('/task/inputs/joy-workloads.json'))
    p.add_argument('--profile', action='store_true')
    p.add_argument('--quality', choices=['exact', 'lossless', 'high'], default='lossless')
    a = p.parse_args()
    contract_path = a.contract
    contract = json.loads(contract_path.read_text())
    image_path = Path('/task') / contract['reference_image']['path']
    image_data = image_path.read_bytes()
    assert hashlib.sha256(image_data).hexdigest() == contract['reference_image']['sha256']
    out = Path('/task/outputs/qualification') / f'{a.arm}-joy-{a.label}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'contract.json').write_bytes(contract_path.read_bytes())
    prompts = contract['prompts'] + [contract['prompts'][0]]
    if a.profile:
        assert a.arm == 'sgl'
        prompts = prompts[:1]
    rows = []
    for index, prompt in enumerate(prompts):
        target = out / f'{index:02d}-{prompt["id"]}'
        target.mkdir()
        payload = {k: str(contract[k]) for k in ['seed', 'num_inference_steps', 'guidance_scale', 'negative_prompt', 'generator_device']}
        payload.update(model=contract['model_path'], prompt=prompt['prompt'], n='1',
                       size=f'{contract["width"]}x{contract["height"]}', response_format='b64_json', output_format='png')
        if a.arm == 'sgl':
            payload.update(quality=a.quality, perf_dump_path=str(target / 'perf.json'))
            if a.profile:
                payload.update(profile='true', num_profiled_timesteps='3')
        (target / 'request.json').write_text(json.dumps({'form': payload, 'image': contract['reference_image']}, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'request': str(target), 'status': 'submitting'}), flush=True)
        start = time.monotonic()
        port = 31040 if a.arm == 'sgl' else 31041
        response = requests.post(f'http://127.0.0.1:{port}/v1/images/edits', data=payload,
                                 files={'image': ('reference.png', image_data, 'image/png')}, timeout=1800)
        elapsed = time.monotonic() - start
        (target / 'response.json').write_text(response.text)
        response.raise_for_status()
        data = response.json()
        assert len(data['data']) == 1
        content = base64.b64decode(data['data'][0]['b64_json'])
        (target / 'output.png').write_bytes(content)
        row = dict(request=str(target), http_seconds=elapsed, bytes=len(content),
                   sha256=hashlib.sha256(content).hexdigest(), benchmark=False, profiled=a.profile)
        rows.append(row)
        (out / 'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    if not a.profile:
        print(json.dumps({'repeat_short_byte_identical': rows[0]['sha256'] == rows[-1]['sha256']}), flush=True)


if __name__ == '__main__':
    main()
