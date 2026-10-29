"""Submit frozen native Anima requests; retain unmodified PNGs."""
import argparse
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

import requests
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['sgl', 'omni'])
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--quality', choices=['exact', 'lossless', 'high'], default='lossless')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    contract_bytes = args.contract.read_bytes()
    contract = json.loads(contract_bytes)
    assert contract['batch'] == contract['concurrency'] == 1
    assert contract['physical_gpus'] == [0]
    assert contract['task'] == 't2i' and contract['num_inference_steps'] == 50
    if args.profile:
        assert args.arm == 'sgl'

    port = 31050 if args.arm == 'sgl' else 31051
    source = Path('/task/repos') / ('sglang' if args.arm == 'sgl' else 'vllm-omni')
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--'])
    expected = ('af21e2b989e482cfdce39689126562f2209d53ac' if args.arm == 'sgl'
                else '817d4c17feb702289a2d2b2cbacfd594c262b7a7')
    assert head == expected and not patch, 'Initial qualification requires clean frozen source'
    out = Path('/task/outputs/qualification') / f'{args.arm}-anima-{args.label}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'contract.json').write_bytes(contract_bytes)
    (out / 'metadata.json').write_text(json.dumps(dict(
        arm=args.arm, source_sha=head, source_diff_sha256=hashlib.sha256(patch).hexdigest(),
        input_sha256=hashlib.sha256(contract_bytes).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        benchmark=False, profiled=args.profile, port=port), indent=2)+'\n')
    prompts = contract['prompts'] + [contract['prompts'][0]]
    if args.profile:
        prompts = prompts[:1]
    rows = []
    for position, prompt in enumerate(prompts):
        target = out / f'{position:02d}-{prompt["id"]}'
        target.mkdir()
        payload = {key: contract[key] for key in
                   ('seed', 'guidance_scale', 'negative_prompt', 'generator_device',
                    'num_inference_steps', 'max_sequence_length')}
        payload.update(model=contract[args.arm + '_model_path'], prompt=prompt['prompt'], n=1,
                       size=f'{contract["width"]}x{contract["height"]}',
                       response_format='b64_json', output_format='png',
                       output_compression=contract['png_compression'][args.arm + '_output_compression'])
        if args.arm == 'omni':
            # Images API has no max_sequence_length field; native Anima default is 512.
            payload.pop('max_sequence_length')
        if args.arm == 'sgl':
            payload.update(quality=args.quality, perf_dump_path=str(target/'perf.json'))
            if args.profile:
                payload.update(profile=True, num_profiled_timesteps=3)
        (target/'request.json').write_text(json.dumps(dict(endpoint='generations', payload=payload),
            ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(dict(request=str(target), status='submitting')), flush=True)
        started = time.monotonic()
        response = requests.post(f'http://127.0.0.1:{port}/v1/images/generations', json=payload, timeout=1800)
        elapsed = time.monotonic() - started
        (target/'response.json').write_text(response.text)
        response.raise_for_status()
        body = response.json()
        assert len(body['data']) == 1
        png = base64.b64decode(body['data'][0]['b64_json'], validate=True)
        output = target/'output.png'
        output.write_bytes(png)
        with Image.open(output) as image:
            image.load()
            assert image.format == 'PNG' and image.mode == 'RGB' and image.size == (contract['width'], contract['height'])
            row = dict(request=str(target), http_seconds=elapsed, bytes=len(png),
                       sha256=hashlib.sha256(png).hexdigest(), mode=image.mode,
                       bands=image.getbands(), channel_extrema=image.getextrema(),
                       benchmark=False, profiled=args.profile)
        rows.append(row)
        (out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    print(json.dumps(dict(repeated_short_byte_identical=rows[0]['sha256']==rows[-1]['sha256'],
                          repeated_short_tested=not args.profile)), flush=True)


if __name__ == '__main__':
    main()
