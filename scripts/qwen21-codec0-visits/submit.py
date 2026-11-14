"""Native Qwen requests with full-step receipts and lossless RGBA provenance."""
import base64
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

arm, run = sys.argv[1:]
assert arm in ('sgl', 'omni') and Path(run).name == run
folder = Path('/task/runs') / run
visit = json.loads((folder / 'visit.json').read_text())
contract = json.loads(Path('/task/inputs/qwen21-native-workloads.json').read_text())
model = next(x for x in contract['models'] if x['model'] == visit['model'])
assert model['revision'] == visit['weights_revision']
width, height = visit.get('image_size', [contract['width'], contract['height']])
assert all(isinstance(x, int) and 0 < x <= 2048 and x % 32 == 0 for x in (width, height))
steps = model['expected_iterations']
reference = Path(contract['edit']['input_path'])
reference_bytes = reference.read_bytes()
assert hashlib.sha256(reference_bytes).hexdigest() == contract['edit']['sha256']
out = folder / 'outputs'
out.mkdir(exist_ok=False)
session = requests.Session()
session.trust_env = False
base = 'http://127.0.0.1:' + ('32150' if arm == 'sgl' else '32151')
rows = []
try:
    for index, item in enumerate(visit['requests']):
        prompt_id = item['prompt_id']
        edit = item['task'] == 'edit'
        assert item['task'] in ('t2i', 'edit')
        directory = out / f'{index:02d}-{prompt_id}'
        directory.mkdir()
        prompt = contract['edit']['prompt'] if edit else next(
            p['prompt'] for p in contract['prompts'] if p['id'] == prompt_id)
        payload = {k: contract[k] for k in (
            'seed', 'generator_device', 'guidance_scale', 'true_cfg_scale', 'negative_prompt')}
        if model['num_inference_steps'] is not None:
            payload['num_inference_steps'] = model['num_inference_steps']
        payload.update(prompt=prompt, n=1, size=f'{width}x{height}', response_format='b64_json',
                       output_format='png', output_compression=100)
        payload['output_compression'] = visit['output_compression'][arm]
        if arm == 'sgl':
            quality = item.get('quality', visit.get('quality', 'lossless'))
            assert quality in ('exact', 'lossless', 'high')
            payload.update(quality=quality, perf_dump_path=str(directory / 'perf.json'))
        else:
            payload['return_stage_metrics'] = True
        do_profile = bool(item.get('profile'))
        if do_profile:
            assert arm == 'sgl' and item['role'] != 'measure'
            payload.update(profile=True, profile_all_stages=True, num_profiled_timesteps=steps)
        (directory / 'request.json').write_text(json.dumps(payload, indent=2)+'\n')
        if edit:
            (directory / 'reference.json').write_text(json.dumps(contract['edit'], indent=2)+'\n')
        log_start = (folder / 'server.log').stat().st_size
        started_wall_ns = time.time_ns()
        started = time.perf_counter_ns()
        if edit:
            form = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in payload.items()}
            response = session.post(base + '/v1/images/edits', data=form,
                files={'image': ('reference.png', reference_bytes, 'image/png')}, timeout=3600)
        else:
            response = session.post(base + '/v1/images/generations', json=payload, timeout=3600)
        received = time.perf_counter_ns()
        (directory / 'response.json').write_bytes(response.content)
        response.raise_for_status()
        result = response.json()
        assert len(result['data']) == 1
        content = base64.b64decode(result['data'][0]['b64_json'], validate=True)
        (directory / 'output.png').write_bytes(content)
        materialized = time.perf_counter_ns()
        image = Image.open(io.BytesIO(content))
        image.load()
        assert image.format == 'PNG' and image.size == (width, height)
        rgba = image.convert('RGBA')
        (directory / 'response-headers.json').write_text(json.dumps(dict(response.headers), indent=2)+'\n')
        with (folder / 'server.log').open('rb') as log:
            log.seek(log_start)
            request_log = log.read().decode(errors='replace')
        (directory / 'server-request.log').write_text(request_log)
        assert f'{steps}/{steps}' in request_log, 'native full-step completion missing'
        if arm == 'sgl':
            perf = json.loads((directory / 'perf.json').read_text())
            assert len(perf['denoise_steps_ms']) == steps
        digest = hashlib.sha256(content).hexdigest()
        expected = item.get('expected_sha256', {}).get(arm)
        row = dict(index=index, prompt_id=prompt_id, task=item['task'], role=item['role'],
            image_size=[width, height], quality=payload.get('quality', 'native_default'),
            benchmark=item['role'] == 'measure', profiler=do_profile,
            native_steps_observed=steps, start_wall_ns=started_wall_ns,
            timer='Before synchronous POST through raw JSON write and base64 decode/PNG byte write; PIL pixel decode, validation and hashes excluded; no fsync',
            timer_ns=dict(start=started, received=received, materialized=materialized),
            saved_request_e2e_seconds=(materialized-started)/1e9,
            bytes=len(content), sha256=digest, original_mode=image.mode,
            decoded_rgba_sha256=hashlib.sha256(rgba.tobytes()).hexdigest(),
            alpha_extrema=rgba.getchannel('A').getextrema(),
            expected_sha256=expected, matches_reference=None if expected is None else digest == expected,
            output=str(directory / 'output.png'))
        rows.append(row)
        (out / 'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        assert expected is None or digest == expected, 'Native output changed; preserve and investigate'
finally:
    session.close()
