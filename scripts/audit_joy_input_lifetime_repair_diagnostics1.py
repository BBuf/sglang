"""Audit complete diagnostic source, request, raw media and profile chains locally."""
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from analyze_joy_gelu_profiles import analyze

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'evidence'
RUN = 'joy-input-lifetime-repair1-diagnostics1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


plan_path = EVIDENCE/'plans'/(RUN+'.json')
plan = read(plan_path)
archive_receipt = EVIDENCE/'checkpoints'/(RUN+'-local-verification.json')
verified = read(archive_receipt)
assert verified['all_member_hashes_verified'] and verified['all_members_extracted']
reproduction = EVIDENCE/'reproduction'/RUN
assert digest(reproduction/(RUN+'.json')) == digest(plan_path)
for path, expected in plan['sha256'].items():
    relative = Path(path)
    if relative.parts[0] == 'scripts':
        archived = reproduction/relative.name
    elif relative.parts[0] == 'outputs':
        archived = EVIDENCE/relative.relative_to('outputs')
    else:
        archived = ROOT/relative
    assert digest(archived) == expected, path
status = read(reproduction/(RUN+'-status.json'))
assert status['status'] == 'raw_complete_archive_and_local_audit_pending'
assert len(status['completed']) == 6 and status['outputs'] == plan['expected_outputs']
for path in plan['required_admissions']:
    assert read(ROOT/path)['passed'], path
test = read(EVIDENCE/'preflight/joy-input-lifetime-repair1-tests-admission.json')
assert test['source_sha'] == plan['sources']['current']['head']
assert test['tests_passed'] == 1 and test['tests_skipped'] == 0
assert test['red_before_fix'] and test['green_after_fix']
for phase, expected_code in [('red-attempt2', 1), ('green-attempt1', 0)]:
    folder = EVIDENCE/'diagnostics'/('joy-input-lifetime-repair1-'+phase)
    receipt = read(folder/'receipt.json')
    key = 'red_receipt_sha256' if expected_code else 'green_receipt_sha256'
    assert digest(folder/'receipt.json') == test[key]
    assert receipt['expected_result_observed'] and receipt['returncode'] == expected_code
    for name, expected in receipt['files_sha256'].items():
        assert digest(folder/name) == expected
    cases = list(ET.parse(folder/'junit.xml').getroot().iter('testcase'))
    assert len(cases) == 1 and cases[0].find('skipped') is None and cases[0].find('error') is None
    if expected_code:
        assert cases[0].find('failure') is not None
        assert 'Joy MLP retained its temporary input' in ''.join(cases[0].find('failure').itertext())
    else:
        assert cases[0].find('failure') is None
checkpoint = read(EVIDENCE/'preflight/joy-current-head-tests1-admission.json')
assert checkpoint['passed'] and checkpoint['checkpoint_files_rehashed'] == 36
assert checkpoint['checkpoint_manifest_sha256'] == plan['sha256']['evidence/downloads/joy-modelscope-download-manifest.json']
chains, profiles, quality_rows, stages = [], {}, [], []
outputs = 0
for shape in plan['shapes']:
    contract_path = ROOT/Path(plan['contracts'][shape]).relative_to('/task')
    contract = read(contract_path)
    for source in plan['source_order']:
        name = RUN+'-'+shape+'-'+source
        block = EVIDENCE/'block-runs'/name
        meta = read(block/'metadata.json')
        assert meta['source_sha'] == plan['sources'][source]['head']
        assert meta['source_diff_sha256'] == digest(block/'source.patch') == hashlib.sha256(b'').hexdigest()
        assert meta['plan_sha256'] == digest(plan_path)
        assert meta['synchronized_stages'] and not meta['primary_timing']
        assert meta['status'] == 'raw_quality_and_diagnostics_complete_local_audit_pending'
        assert digest(block/'launch.sh') == plan['sha256']['scripts/'+plan['launcher']]
        assert 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1' in (block/'launch.sh').read_text()
        assert not any(marker in (block/'server.log').read_text(errors='replace') for marker in plan['forbidden_backend_markers'])
        expected_phases = [(str(i), quality, False) for i,quality in enumerate(plan['quality_order'])]
        if shape in plan['profile_shapes']:
            expected_phases.append(('profile', 'high', True))
        assert len(meta['qualifications']) == len(expected_phases)
        for record, (phase, quality, profiled) in zip(meta['qualifications'], expected_phases):
            assert (record['phase'], record['quality'], record['profiled']) == (phase, quality, profiled)
            folder = EVIDENCE/'qualification'/('sgl-joy-'+name+'-'+phase+'-'+quality)
            assert digest(folder/'contract.json') == digest(contract_path)
            rows = read(folder/'results.json')
            reference = EVIDENCE/'qualification'/plan['references'][shape][source][quality]
            refs = read(reference/'results.json')
            prompts = contract['prompts']+[contract['prompts'][0]]
            count = 1 if profiled else 4
            assert len(rows) == record['outputs'] == count
            raw_stages = []
            for row, old, prompt in zip(rows, refs, prompts):
                request_folder = folder/Path(row['request']).name
                assert digest(request_folder/'output.png') == row['sha256'] == old['sha256']
                assert digest(reference/Path(old['request']).name/'output.png') == old['sha256']
                assert row['profiled'] == profiled and not row['benchmark']
                request = read(request_folder/'request.json')
                assert request['image'] == contract['reference_image']
                form = request['form']
                for key in ('seed', 'num_inference_steps', 'guidance_scale', 'negative_prompt', 'generator_device'):
                    assert form[key] == str(contract[key])
                assert form['size'] == f'{contract["width"]}x{contract["height"]}'
                assert form['model'] == contract['model_path'] and form['quality'] == quality
                assert form['prompt'] == prompt['prompt'] and form['n'] == '1'
                assert (form.get('profile') == 'true') == profiled
                if profiled:
                    assert form['num_profiled_timesteps'] == '3'
                else:
                    perf = read(request_folder/'perf.json')
                    seconds = next(s['duration_ms']/1000 for s in perf['steps'] if s['name'] == 'DenoisingStage')
                    raw_stages.append(dict(request=request_folder.name, denoise_seconds=seconds, http_seconds=row['http_seconds']))
                    if source == 'current':
                        main = EVIDENCE/'qualification'/('sgl-joy-'+RUN+'-'+shape+'-main-'+phase+'-'+quality)/request_folder.name/'output.png'
                        a = np.asarray(Image.open(main).convert('RGB'))
                        b = np.asarray(Image.open(request_folder/'output.png').convert('RGB'))
                        exact = bool(np.array_equal(a,b))
                        ssim = float(structural_similarity(a,b,channel_axis=2,data_range=255))
                        psnr = None if exact else float(peak_signal_noise_ratio(a,b,data_range=255))
                        passed = exact if quality == 'lossless' else exact or (ssim >= .95 and psnr >= 28)
                        quality_rows.append(dict(shape=shape, phase=phase, quality=quality, request=request_folder.name,
                            pixel_identical=exact, ssim=ssim, psnr_db=psnr, psnr_infinite=exact, passed=passed,
                            main_sha256=digest(main), current_sha256=row['sha256']))
                outputs += 1
            assert raw_stages == record['stages']
            if quality == 'high' and not profiled:
                p50 = statistics.median(raw_stages[i]['denoise_seconds'] for i in (0,3))
                assert meta['high_short_denoise_p50_seconds'] == p50
                stages.append(dict(shape=shape, source=source, n_short=2, denoise_p50_seconds=p50,
                                   launch_to_ready_seconds=meta['launch_to_ready_seconds'], primary_timing=False))
        if shape in plan['profile_shapes']:
            trace_meta = status['profiles'][shape+'-'+source]
            trace = EVIDENCE/Path(trace_meta['path']).relative_to('outputs')
            assert digest(trace) == trace_meta['sha256'] and trace.stat().st_size == trace_meta['bytes']
            actual = analyze(trace, 3)
            recorded = read(block/'profile-analysis.json')
            assert {k:v for k,v in actual.items() if k != 'trace'} == {k:v for k,v in recorded.items() if k != 'trace'}
            assert actual['cuda_graph_launch_count'] == 0
            profiles[shape+'-'+source] = {k:actual[k] for k in ('sha256','kernel_count','kernel_sum_ms','gelu_cpu_calls','fused_custom_op_calls','activation_calls')}
        chains.append(dict(shape=shape, source=source, source_sha=meta['source_sha'], metadata_sha256=digest(block/'metadata.json')))
    if shape in plan['profile_shapes']:
        main, current = profiles[shape+'-main'], profiles[shape+'-current']
        assert main['gelu_cpu_calls']-current['gelu_cpu_calls'] == 80
        assert current['fused_custom_op_calls'] == current['activation_calls'] == 80
assert outputs == plan['expected_outputs'] == 76
assert len(quality_rows) == 36
result = dict(passed=all(row['passed'] for row in quality_rows), source_sha=plan['sources']['current']['head'],
    plan_sha256=digest(plan_path), archive_verification_sha256=digest(archive_receipt),
    all_raw_outputs_rehashed=outputs, quality_rows=quality_rows, profiles=profiles,
    synchronized_diagnostic_stages=stages, source_chains=chains, actual_SM120_regression_passed=True,
    primary_timing=False, final_PR_admitted=False,
    caveats=['Lossless-high-lossless in one process per shape/source; short/long/Chinese/repeated short each phase.',
             'Two synchronized short stage samples per diagnostic process; never pool these HTTP latencies with SYNC0 primary timings.',
             'Tall is quality/shape compatibility only. No new BCG, compiled GPU, full-model LoRA or exact-mode performance claim.',
             'Actual main/current visual review and publication decision remain separate gates. No new optimization slot is assigned.'])
out = EVIDENCE/'preflight'/(RUN+'-admission.json')
assert not out.exists()
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('quality_rows','source_chains')},indent=2))
