"""Audit separate native Omni stage/precision diagnostics; never emit E2E speedups."""
import argparse
import base64
import hashlib
import json
import statistics
from pathlib import Path

from analyze_joy_blocks import digest, read
from inspect_joy_native_diagnostic import inspect


def audit(shape):
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'evidence'
    plan_path = evidence / 'benchmarks' / f'joy-gelu-final-{shape}-omni-denoise1-plan.json'
    plan = read(plan_path)
    assert plan['diagnostic_profiler'] and not plan['primary_performance_evidence']
    assert plan['order'] == ['omni_compile_profile']
    name = plan['id'] + '-00-omni_compile_profile'
    folder = evidence / 'benchmarks' / name
    server_folder = evidence / 'block-runs' / name
    metadata, server = read(folder / 'metadata.json'), read(server_folder / 'metadata.json')
    assert server['plan'] == plan and server['index'] == 0 and server['config'] == plan['order'][0]
    assert metadata['arm'] == 'omni' and metadata['profiler']
    assert metadata['args']['diagnostic_profiler']
    assert metadata['physical_gpus'] == [0]
    assert metadata['source_sha'] == '817d4c17feb702289a2d2b2cbacfd594c262b7a7'
    assert metadata['source_diff_sha256'] == hashlib.sha256(b'').hexdigest()
    assert not (folder / 'source-status.txt').read_text().strip()
    contract_path = root / Path(plan['contract']).relative_to('/task')
    assert digest(contract_path) == metadata['input_sha256'] == plan['frozen_input_sha256']
    assert metadata['args']['contract'] == plan['contract']
    for key in ('warmups', 'repeats', 'quality'):
        assert metadata['args'][key] == plan[key]
    contract = read(contract_path)
    assert digest(root / contract['reference_image']['path']) == contract['reference_image']['sha256']
    hook = root / 'scripts/joy_precision_sitecustomize.py'
    assert digest(hook) == plan['precision_observer']['sha256']
    launcher = (server_folder / 'launch.sh').read_text()
    assert '--enable-diffusion-pipeline-profiler' in launcher
    assert 'JOY_PRECISION_AUDIT_DIR=/task/outputs/precision/omni-native' in launcher

    def audit_request(request, prompt):
        for key in ('seed', 'num_inference_steps', 'guidance_scale', 'negative_prompt', 'generator_device'):
            assert request[key] == str(contract[key])
        assert request['prompt'] == prompt and request['model'] == contract['model_path']
        assert request['size'] == f'{contract["width"]}x{contract["height"]}'
        assert request['n'] == '1' and request['output_format'] == 'png'

    qualified_name = 'omni-joy-' + name
    qualified = evidence / 'qualification' / qualified_name
    quality = read(qualified / 'results.json')
    expected_ids = [f'{i:02d}-{p["id"]}' for i, p in enumerate(contract['prompts'])]
    expected_ids.append('03-' + contract['prompts'][0]['id'])
    assert len(quality) == 4
    quality_by_prompt = {}
    for i, row in enumerate(quality):
        prompt_id = Path(row['request']).name
        assert prompt_id == expected_ids[i]
        output = qualified / prompt_id / 'output.png'
        assert digest(output) == row['sha256']
        request = read(output.parent / 'request.json')
        assert request['image'] == contract['reference_image']
        audit_request(request['form'], contract['prompts'][i % 3]['prompt'])
        quality_by_prompt[prompt_id] = row
    gate = read(evidence / 'qualification' / (qualified_name + '-quality.json'))
    references = plan['quality_references']['omni_compile_profile']
    assert gate['all_pass'] and len(gate['results']) == 4 * len(references)
    assert {(x['reference'], x['prompt']) for x in gate['results']} == {
        (reference, prompt) for reference in references for prompt in quality_by_prompt}
    for row in gate['results']:
        assert row['passes_cross_framework_gate']
        assert row['candidate_sha256'] == quality_by_prompt[row['prompt']]['sha256']
        reference = evidence / 'qualification' / row['reference'] / row['prompt'] / 'output.png'
        assert digest(reference) == row['reference_sha256']

    records = read(folder / 'results.json')
    assert len(records) == plan['warmups'] + plan['repeats']
    measured_gate = read(folder / 'quality.json')
    assert measured_gate['all_pass'] and len(measured_gate['rows']) == len(records)
    reference = evidence / Path(plan['omni_short_quality_reference']).relative_to('/task/outputs')
    assert measured_gate['reference'] == plan['omni_short_quality_reference']
    assert digest(reference) == measured_gate['reference_sha256']
    allowed = set(metadata['args']['allowed_gpu_pids'])
    assert allowed == set(server['gpu_container_pids'][0]) and not server['gpu_container_pids'][1]
    uuid = ['GPU-1565b45e-a702-2015-9776-61f68c36fd59', 'GPU-7ca33735-71c8-fc89-57a6-c7cdbf8d28cd']
    correlation, _ = inspect(evidence, shape, archived=True)
    seconds = []
    for i, (row, numerical) in enumerate(zip(records, measured_gate['rows'])):
        request_folder = folder / f'{i:02d}'
        assert row['index'] == numerical['index'] == i
        assert row['warmup'] == numerical['warmup'] == (i < plan['warmups'])
        assert digest(request_folder / 'output.png') == row['sha256'] == numerical['candidate_sha256']
        assert numerical['pass_gate']
        assert numerical['pixel_exact'] or (numerical['ssim'] >= .95 and numerical['psnr_db'] >= 28)
        audit_request(read(request_folder / 'request.json'), contract['prompts'][0]['prompt'])
        response = read(request_folder / 'response.json')
        assert hashlib.sha256(base64.b64decode(response['data'][0]['b64_json'])).hexdigest() == row['sha256']
        value = correlation['correlations'][5+i]['native_diffuse_seconds']
        assert 0 < value < row['http_seconds']
        if not row['warmup']:
            seconds.append(value)
        samples = read(request_folder / 'gpu-samples.json')
        assert len(samples) >= 2
        for sample in samples:
            g0, g1 = sample['gpus']
            assert [g0['uuid'], g1['uuid']] == uuid
            assert set(g0['pids']) <= allowed and not g1['pids']

    precision_path = folder / 'runtime-precision.jsonl'
    all_precision = [json.loads(line) for line in precision_path.read_text().splitlines()]
    assert len(all_precision) == 5 + len(records)
    precision = all_precision[1:]
    assert len(all_precision[0]['scheduler_calls']) == 2
    assert correlation['native_startup_records'] == 1
    assert len({row['pid'] for row in precision}) == 1
    scheduler_signatures = set()
    request_signatures = set()
    last_finished = 0
    for row in precision:
        assert row['hook_sha256'] == digest(hook)
        assert row['name'] == plan['required_stage_metric']
        assert last_finished < row['started_ns'] < row['finished_ns']
        last_finished = row['finished_ns']
        assert len(row['scheduler_calls']) == contract['num_inference_steps'] == 30
        request_signatures.add(json.dumps(dict(inputs=row['inputs'], output=row['output'],
                                               cuda_autocast=row['cuda_autocast']), sort_keys=True))
        for call in row['scheduler_calls']:
            for key in ('model_output', 'sample', 'result'):
                assert call[key]['device'].startswith('cuda')
                assert call[key]['dtype'] in ('torch.bfloat16', 'torch.float32')
                assert all(n > 0 for n in call[key]['shape'])
            assert call['model_output']['shape'] == call['sample']['shape'] == call['result']['shape']
            scheduler_signatures.add(json.dumps(call, sort_keys=True))
    result = dict(shape=shape, all_raw_audits_pass=True, primary_timing=False,
        native_denoise_metric=plan['required_stage_metric'], units='seconds',
        metric_source=correlation['metric_source'], native_startup_records_retained=1,
        native_log_correlation=correlation,
        denoise_seconds=seconds, denoise_p50_seconds=statistics.median(seconds),
        diagnostic_requests=len(records), qualification_requests=4, native_step_count=30,
        actual_request_precision_signatures=[json.loads(s) for s in sorted(request_signatures)],
        actual_scheduler_precision_signatures=[json.loads(s) for s in sorted(scheduler_signatures)],
        precision_jsonl_sha256=digest(precision_path), plan_sha256=digest(plan_path),
        observer_sha256=digest(hook),
        caveat='One separate native-profiler process per shape, four measured diagnostic requests after qualification and warmup. Metadata-only scheduler observation adds Python overhead inside this diagnostic stage. Never use these HTTP durations or profiler durations to replace the independent primary E2E comparison. Serial observer rows correspond to four quality, two warmup and four diagnostic requests in controller order; no concurrent requests.')
    target = evidence / 'benchmarks' / f'joy-gelu-final-{shape}-omni-denoise1-audit.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('shape', choices=['square', 'wide'])
    audit(parser.parse_args().shape)
