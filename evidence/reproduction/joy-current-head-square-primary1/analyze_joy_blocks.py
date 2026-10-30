"""Audit all planned Joy serving blocks from archived raw data before comparing."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('plan', type=Path)
    p.add_argument('--evidence', type=Path, default=Path('evidence'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plan = read(a.plan)
    blocks = []
    task_root = Path(__file__).resolve().parents[1]
    remote_contract = Path(plan.get('contract', '/task/inputs/joy-workloads.json'))
    contract_path = task_root / remote_contract.relative_to('/task')
    assert digest(contract_path) == plan['frozen_input_sha256']
    contract = read(contract_path)
    expected_uuids = ['GPU-1565b45e-a702-2015-9776-61f68c36fd59',
                      'GPU-7ca33735-71c8-fc89-57a6-c7cdbf8d28cd']
    empty_sha = hashlib.sha256(b'').hexdigest()
    for index, config in enumerate(plan['order']):
        name = f'{plan["id"]}-{index:02d}-{config}'
        folder = a.evidence / 'benchmarks' / name
        meta = read(folder / 'metadata.json')
        server = read(a.evidence / 'block-runs' / name / 'metadata.json')
        records = read(folder / 'results.json')
        summary = read(folder / 'summary.json')
        assert server['plan'] == plan and server['config'] == config and server['index'] == index
        assert meta['args']['warmups'] == plan['warmups'] and meta['args']['repeats'] == plan['repeats']
        assert meta['physical_gpus'] == [0]
        assert meta['input_sha256'] == plan['frozen_input_sha256']
        assert meta['args']['contract'] == str(remote_contract)
        assert not meta['profiler']
        assert meta['args']['quality'] == plan['quality']
        arm = 'omni' if config.startswith('omni') else 'sgl'
        candidate = config.startswith('candidate')
        assert meta['arm'] == arm
        candidate_source = plan.get('candidate_sources', {}).get(config, plan)
        expected_sha = (candidate_source['candidate_head_sha'] if candidate else
                        '817d4c17feb702289a2d2b2cbacfd594c262b7a7' if arm == 'omni' else
                        'af21e2b989e482cfdce39689126562f2209d53ac')
        expected_diff = candidate_source['candidate_diff_sha256'] if candidate else empty_sha
        assert meta['source_sha'] == expected_sha
        assert meta['source_diff_sha256'] == expected_diff
        if expected_diff != empty_sha:
            assert digest(folder / 'source.patch') == expected_diff
        else:
            status = (folder / 'source-status.txt').read_text().splitlines()
            if status:
                # Prior profiler traces are untracked artifacts, not source edits.
                assert arm == 'sgl' and not candidate and status == ['?? logs/']
                artifacts = read(a.evidence / 'preflight/joy-main-untracked-artifacts.json')
                assert artifacts['head'] == expected_sha and artifacts['tracked_diff_sha256'] == empty_sha
                for artifact in artifacts['untracked']:
                    path = Path(artifact['path'])
                    assert path.parent == Path('logs') and path.name.endswith('.trace.json.gz')
                    archived = a.evidence / 'profiles' / path.name
                    assert digest(archived) == artifact['sha256']
        assert len(records) == plan['warmups'] + plan['repeats']
        assert [r['index'] for r in records] == list(range(len(records)))
        qualified_name = f'{arm}-joy-{name}'
        qualified = a.evidence / 'qualification' / qualified_name
        quality_rows = read(qualified / 'results.json')
        assert len(quality_rows) == 4
        for row in quality_rows:
            request_path = Path(row['request'])
            assert request_path.parent == Path('/task/outputs/qualification') / qualified_name
            assert digest(qualified / request_path.name / 'output.png') == row['sha256']
        if config in plan.get('byte_identical_quality_reference', {}):
            reference = a.evidence/'qualification'/plan['byte_identical_quality_reference'][config]
            reference_rows = read(reference/'results.json')
            assert [r['sha256'] for r in quality_rows] == [r['sha256'] for r in reference_rows]
            for row in reference_rows:
                assert digest(reference/Path(row['request']).name/'output.png') == row['sha256']
        gate = read(a.evidence / 'qualification' / (qualified_name + '-quality.json'))
        assert gate['all_pass'] and len(gate['results']) == 4 * len(plan['quality_references'][config])
        assert {r['reference'] for r in gate['results']} == set(plan['quality_references'][config])
        assert len({(r['reference'], r['prompt']) for r in gate['results']}) == len(gate['results'])
        for row in gate['results']:
            source = qualified / row['prompt'] / 'output.png'
            reference = a.evidence / 'qualification' / row['reference'] / row['prompt'] / 'output.png'
            assert digest(source) == row['candidate_sha256']
            assert digest(reference) == row['reference_sha256']
            assert row['passes_cross_framework_gate']
        if arm == 'sgl':
            assert all(r['sha256'] == quality_rows[0]['sha256'] for r in records)
            assert quality_rows[0]['sha256'] == quality_rows[-1]['sha256']
        else:
            measured_gate = read(folder / 'quality.json')
            assert measured_gate['all_pass'] and len(measured_gate['rows']) == len(records)
            ref_path = Path(plan['omni_short_quality_reference'])
            local_ref = a.evidence / ref_path.relative_to('/task/outputs')
            assert measured_gate['reference'] == str(ref_path)
            assert digest(local_ref) == measured_gate['reference_sha256']
            for result, raw in zip(measured_gate['rows'], records):
                assert result['index'] == raw['index'] and result['warmup'] == raw['warmup']
                assert result['candidate_sha256'] == raw['sha256'] and result['pass_gate']
                assert result['pixel_exact'] or (result['ssim'] >= .95 and result['psnr_db'] >= 28)
        values, denoise, engine, memory, telemetry = [], [], [], [], []
        allowed = set(meta['args']['allowed_gpu_pids'])
        assert allowed == set(server['gpu_container_pids'][0]) and not server['gpu_container_pids'][1]
        for row in records:
            request_folder = folder / f'{row["index"]:02d}'
            request = read(request_folder / 'request.json')
            for key in ['seed', 'num_inference_steps', 'guidance_scale', 'negative_prompt', 'generator_device']:
                assert request[key] == str(contract[key])
            assert request['prompt'] == contract['prompts'][0]['prompt']
            assert request['model'] == contract['model_path'] and request['n'] == '1'
            assert request['size'] == f'{contract["width"]}x{contract["height"]}'
            if arm == 'sgl':
                assert request['quality'] == plan['quality'] and request['profile'] == 'false'
            assert digest(request_folder / 'output.png') == row['sha256']
            assert row['warmup'] == (row['index'] < plan['warmups'])
            samples = read(request_folder / 'gpu-samples.json')
            assert len(samples) >= 2
            for sample in samples:
                g0, g1 = sample['gpus']
                assert [g0['uuid'], g1['uuid']] == expected_uuids
                assert set(g0['pids']) <= allowed and not g1['pids']
            if row['warmup']:
                continue
            values.append(row['http_seconds'])
            memory.append(max(s['gpus'][0]['memory_bytes'] for s in samples))
            telemetry.extend(s['gpus'][0] for s in samples)
            if arm == 'sgl':
                perf = read(request_folder / 'perf.json')
                denoise.append(next(s['duration_ms'] / 1000 for s in perf['steps'] if s['name'] == 'DenoisingStage'))
                engine.append(perf['total_duration_ms'] / 1000)
            else:
                response = read(request_folder / 'response.json')
                engine.append(response['metrics']['stage_durations']['stage_0_gen_ms'] / 1000)
        assert len(values) == plan['repeats']
        assert summary['p50_seconds'] == statistics.median(values)
        blocks.append(dict(index=index, config=config, name=name,
            source_sha=meta['source_sha'], source_diff_sha256=expected_diff,
            http_seconds=values, p50_seconds=statistics.median(values), p95_seconds=percentile(values, .95),
            cv_percent=100*statistics.stdev(values)/statistics.mean(values),
            denoise_seconds=denoise, denoise_p50_seconds=statistics.median(denoise) if denoise else None,
            engine_total_p50_seconds=statistics.median(engine),
            sequential_requests_per_second=len(values)/sum(values), peak_gpu_memory_bytes=max(memory),
            launch_to_ready_seconds=server['launch_to_ready_seconds'],
            warmup_http_seconds=[r['http_seconds'] for r in records if r['warmup']],
            prebenchmark_quality_http_seconds=[r['http_seconds'] for r in quality_rows],
            startup_scope='Fresh server with existing disk/compiler caches; four quality requests then explicit edit warmups before primary timings. No cold-cache startup claim.',
            quality_requests_before_warmup=4, ownership_and_output_audit_pass=True,
            sm_clock_mhz_range=[min(s['sm_clock_mhz'] for s in telemetry),max(s['sm_clock_mhz'] for s in telemetry)],
            temperature_c_range=[min(s['temperature_c'] for s in telemetry),max(s['temperature_c'] for s in telemetry)]))
    result = dict(plan=plan, blocks=blocks, all_planned_blocks_present=True,
                  scope=(f'Frozen Joy {contract["width"]}x{contract["height"]}, '
                         f'{contract["num_inference_steps"]} steps, CFG {contract["guidance_scale"]}, '
                         f'precision {contract["precision"]}, GPU0, batch1, concurrency1, {plan["quality"]}. '
                         'Short prompt primary timing; long/Chinese quality only. Not full matrix.'),
                  source_and_ownership_audit_pass=True)
    if 'aa_acceptance' in plan:
        assert len({b['config'] for b in blocks}) == 1
        medians = [b['p50_seconds'] for b in blocks]
        relative = max(medians)/min(medians)-1
        within = max(b['cv_percent'] for b in blocks)
        gate = plan['aa_acceptance']
        result['aa'] = dict(block_median_relative_difference=relative, max_within_block_cv_percent=within,
            acceptance=gate, passed=relative <= gate['max_block_median_relative_difference'] and within <= gate['max_within_block_cv_percent'])
    if 'pair_indices' in plan:
        result['paired'] = {}
        for label, pairs in plan['pair_indices'].items():
            ratios = [blocks[i]['p50_seconds']/blocks[j]['p50_seconds'] for i,j in pairs]
            result['paired'][label] = dict(pair_indices=pairs, ratios=ratios,
                geometric_mean=math.exp(statistics.mean(math.log(x) for x in ratios)), all_pairs_faster=all(x>1 for x in ratios))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('plan','blocks')},indent=2))
    for block in blocks:
        print(json.dumps({k:block[k] for k in ['index','config','p50_seconds','denoise_p50_seconds','cv_percent']}))


if __name__ == '__main__':
    main()
