"""Recompute matched final-head trace evidence and check its source/input/PNG chain."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_joy_blocks import digest, read
from analyze_joy_gelu_profiles import analyze


def audit(shape):
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'evidence'
    contract_path = root / 'inputs' / ('joy-workloads' + ('-wide' if shape == 'wide' else '') + '.json')
    contract = read(contract_path)
    analysis_path = evidence / 'profiles' / f'joy-gelu-final-{shape}-matched-profile-analysis.json'
    recorded = read(analysis_path)
    recomputed = {}
    chains = []
    for arm, key in [('main', 'main'), ('pr', 'candidate')]:
        name = f'joy-gelu-final-{shape}-profile1-{arm}'
        server_folder = evidence / 'block-runs' / name
        metadata = read(server_folder / 'metadata.json')
        head = ('af21e2b989e482cfdce39689126562f2209d53ac' if arm == 'main'
                else '4c135ea92d6b4cbb0ead69bb39976f7dbe995455')
        assert metadata['source_sha'] == head
        assert metadata['diff_sha256'] == digest(server_folder / 'source.patch') == hashlib.sha256(b'').hexdigest()
        assert metadata['contract_sha256'] == digest(contract_path)
        assert metadata['contract'] == '/task/' + str(contract_path.relative_to(root))
        assert metadata['status'] == 'profile_generated_not_primary_timing' and not metadata['benchmark']
        assert len(metadata['traces']) == 1
        trace_record = metadata['traces'][0]
        trace = evidence / 'profiles' / name / Path(trace_record['path']).name
        assert digest(trace) == trace_record['sha256'] and trace.stat().st_size == trace_record['bytes']
        qualified = evidence / 'qualification' / f'sgl-{name}-high'
        assert digest(qualified / 'contract.json') == digest(contract_path)
        rows = read(qualified / 'results.json')
        assert len(rows) == 1 and rows[0]['profiled'] and not rows[0]['benchmark']
        request_folder = qualified / Path(rows[0]['request']).name
        assert digest(request_folder / 'output.png') == rows[0]['sha256']
        reference_name = ('sgl-joy-gelu-v1-main-high' if shape == 'square' else 'sgl-joy-gelu-v4-wide-main-high') if arm == 'main' else f'sgl-joy-gelu-v4-{shape}-pr-high'
        reference = evidence / 'qualification' / reference_name
        reference_row = read(reference / 'results.json')[0]
        assert rows[0]['sha256'] == reference_row['sha256'] == digest(reference / Path(reference_row['request']).name / 'output.png')
        request = read(request_folder / 'request.json')
        assert request['image'] == contract['reference_image']
        form = request['form']
        for field in ('seed', 'num_inference_steps', 'guidance_scale', 'negative_prompt', 'generator_device'):
            assert form[field] == str(contract[field])
        assert form['prompt'] == contract['prompts'][0]['prompt']
        assert form['size'] == f'{contract["width"]}x{contract["height"]}'
        assert form['quality'] == 'high' and form['profile'] == 'true' and form['num_profiled_timesteps'] == '3'
        actual = analyze(trace, 3)
        # The local archived trace path is expected to differ from the remote path.
        assert {k:v for k,v in actual.items() if k != 'trace'} == {k:v for k,v in recorded[key].items() if k != 'trace'}
        recomputed[key] = actual
        chains.append(dict(arm=arm, source_sha=head, trace_sha256=digest(trace),
                           source_metadata_sha256=digest(server_folder / 'metadata.json'),
                           output_sha256=rows[0]['sha256']))
    main, candidate = recomputed['main'], recomputed['candidate']
    assert main['gelu_cpu_calls'] - candidate['gelu_cpu_calls'] == 80
    assert candidate['fused_custom_op_calls'] == candidate['activation_calls'] == 80
    assert all('cublas' in row['name'].lower() or 'gemm' in row['name'].lower()
               for row in candidate['activation_correlated_kernels'])
    assert main['cuda_graph_launch_count'] == candidate['cuda_graph_launch_count'] == 0
    result = dict(shape=shape, final_head_and_raw_trace_audit_pass=True, primary_timing=False,
        source_chains=chains, main=main, candidate=candidate,
        standalone_gelu_calls_removed=80,
        caveat='One separate fresh profiled process per arm, matching complete CPU/GPU ProfilerStep#3. Diagnostic kernel attribution only; no profiler-derived end-to-end speedup. GPU idleness was checked before launch and after exit; this trace run does not have the primary timing run\'s continuous NVML ownership sampling.')
    target = evidence / 'profiles' / f'joy-gelu-final-{shape}-profile-audit.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('main', 'candidate')}, indent=2))
    for arm, row in recomputed.items():
        print(json.dumps(dict(arm=arm, **{k:row[k] for k in ('kernel_count', 'kernel_sum_ms', 'gelu_cpu_calls', 'activation_calls')})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('shape', choices=['square', 'wide'])
    audit(parser.parse_args().shape)
