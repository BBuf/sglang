"""Frozen same-input native image requests; these are qualification artifacts."""
import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['sgl', 'omni'])
    parser.add_argument('--label', default='eager')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--quality', choices=['exact', 'lossless', 'high'], default='lossless')
    args = parser.parse_args()
    port = 31030 if args.arm == 'sgl' else 31031
    base = f'http://127.0.0.1:{port}'
    out = Path(f'/task/outputs/qualification/{args.arm}-moss-{args.label}')
    contract = json.loads(Path('/task/inputs/moss-workloads.json').read_text())
    prompts = contract['prompts'] + [contract['prompts'][0]]
    if args.profile:
        assert args.arm == 'sgl'
        prompts = prompts[:1]
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for index, prompt in enumerate(prompts):
        target = out / f'{index:02d}-{prompt["id"]}'
        target.mkdir()
        common = ['seed', 'num_inference_steps', 'guidance_scale', 'flow_shift', 'negative_prompt', 'generator_device']
        payload = {key: contract[key] for key in common}
        payload.update(model='/task/cache/common/MOSS-Image-Base', prompt=prompt['prompt'],
                       size=f'{contract["width"]}x{contract["height"]}', n=1, response_format='b64_json')
        extra = {key: contract[key] for key in ['uncond_skip_layers', 'slg_start', 'slg_stop']}
        if args.arm == 'sgl':
            payload.update(extra)
            payload.update(quality=args.quality, output_format='png', perf_dump_path=str(target / 'perf.json'))
            if args.profile:
                payload.update(profile=True, num_profiled_timesteps=3)
        else:
            payload['extra_params'] = extra
        (target / 'request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'request': str(target), 'status': 'submitting'}), flush=True)
        started = time.monotonic()
        response = requests.post(base + '/v1/images/generations', json=payload, timeout=1800)
        elapsed = time.monotonic() - started
        (target / 'response.json').write_text(response.text)
        response.raise_for_status()
        data = response.json()
        assert len(data['data']) == 1, data.keys()
        content = base64.b64decode(data['data'][0]['b64_json'])
        (target / 'output.png').write_bytes(content)
        row = {'request': str(target), 'http_seconds': elapsed, 'bytes': len(content),
               'sha256': hashlib.sha256(content).hexdigest(), 'benchmark': False}
        results.append(row)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    if not args.profile:
        print(json.dumps({'short_after_other_prompts_byte_identical': results[0]['sha256'] == results[-1]['sha256']}), flush=True)


if __name__ == '__main__':
    main()
