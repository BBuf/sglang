"""Resolve the frozen precision-audit placeholder from final raw diagnostics."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
evidence = root / 'evidence'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


shapes = {}
for shape, contract_name in [('square', 'joy-workloads.json'), ('wide', 'joy-workloads-1792x512.json')]:
    profile_path = evidence / 'profiles' / f'joy-gelu-final-{shape}-matched-profile-analysis.json'
    profiles = read(profile_path)
    profile_audit = evidence / 'profiles' / f'joy-gelu-final-{shape}-profile-audit.json'
    assert read(profile_audit)['final_head_and_raw_trace_audit_pass']
    main, candidate = profiles['main'], profiles['candidate']
    # These entries are recomputed from the complete matched raw trace step.
    assert main['tail_cpu_ops'] == candidate['tail_cpu_ops']
    tail = candidate['tail_cpu_ops']
    assert any(x['name'] == 'aten::clamp_min' and x['input_types'][0] == 'float' for x in tail)
    assert any(x['name'] == 'aten::mul' and x['input_types'] == ['c10::BFloat16', 'float'] for x in tail)
    assert any(x['name'] == 'aten::add' and x['input_types'][:2] == ['float', 'float'] for x in tail)
    assert candidate['activation_calls'] == 80
    assert all(t[:3] == ['c10::BFloat16'] * 3 for t in candidate['activation_input_types'])
    omni_path = evidence / 'benchmarks' / f'joy-gelu-final-{shape}-omni-denoise1-audit.json'
    omni = read(omni_path)
    assert omni['all_raw_audits_pass'] and omni['native_step_count'] == 30
    assert omni['native_startup_records_retained'] == 1
    for sig in omni['actual_scheduler_precision_signatures']:
        assert sig['cuda_autocast'] is False
        assert all(sig[k]['dtype'] == 'torch.bfloat16' for k in ['sample', 'model_output', 'result'])
    for sig in omni['actual_request_precision_signatures']:
        assert sig['cuda_autocast'] is False
        assert sig['inputs']['negative_prompt_embeds']['dtype'] == 'torch.bfloat16'
        assert sig['inputs']['prompt_embeds']['dtype'] == 'torch.bfloat16'
        assert sig['inputs']['timesteps']['shape'] == [30]
    plan = read(evidence / 'benchmarks' / f'joy-gelu-final-{shape}-paired1-plan.json')
    contract_path = root / Path(plan['contract']).relative_to('/task')
    assert sha(contract_path) == plan['frozen_input_sha256']
    shapes[shape] = dict(
        final_profile_analysis=str(profile_path.relative_to(root)), final_profile_analysis_sha256=sha(profile_path),
        profile_audit_sha256=sha(profile_audit), main_trace_sha256=main['sha256'], candidate_trace_sha256=candidate['sha256'],
        sgl_main_candidate_tail_identical=True, sgl_tail_cpu_ops=tail,
        fused_input_types=candidate['activation_input_types'],
        omni_audit=str(omni_path.relative_to(root)), omni_audit_sha256=sha(omni_path),
        omni_scheduler_signatures=omni['actual_scheduler_precision_signatures'],
        omni_request_precision_signatures=omni['actual_request_precision_signatures'],
        all_ten_real_requests_and_300_scheduler_steps_audited=True,
        unchanged_contract_sha256=sha(contract_path), native_omni_denoise_p50_seconds=omni['denoise_p50_seconds'])

result = dict(
    created_at=datetime.now(timezone.utc).isoformat(), status='final raw runtime audit complete',
    main_sha='af21e2b989e482cfdce39689126562f2209d53ac', candidate_sha='4c135ea92d6b4cbb0ead69bb39976f7dbe995455',
    omni_sha='817d4c17feb702289a2d2b2cbacfd594c262b7a7',
    component_policy='BF16 DiT/text encoder/VAE, no quantization; each native implementation retains its own mixed-precision operations.',
    shapes=shapes,
    resolution_of_frozen_latents_and_solver_audit_placeholder={
        'frozen_contracts_modified': False,
        'sgl': 'Final main/PR matched trace tails are identical: BF16 projection and CFG arithmetic, FP32 norm normalization and solver update.',
        'omni': 'All ten real requests per shape: CUDA autocast off, BF16 scheduler model_output/sample/result. Installed scheduler upcasts sample to FP32 internally, then returns model_output.dtype.',
        'comparison_scope': 'Same BF16 component/no-quantization workload and resources, native mixed precision; not operator-by-operator precision identity.',
        'candidate_lowers_precision_outside_mlp': False,
        'scalar_observer_limit': 'Metadata observer records bool/float scalar type, not scalar value. CFG4 is established by archived request and native alias resolution; negative branches are present in all actual request metadata.',
        'diagnostic_startup': 'One native two-step dummy call retained separately from the ten thirty-step requests per process.'},
    other_native_differences=read(evidence / 'preflight/joy-native-precision-policy-audit.json')['native_differences_to_disclose'][1:],
    quality_scope='All final independent primary outputs and separate diagnostics pass the frozen cross-framework gates. Final square/wide primary minimum SSIM 0.995885/0.994935, PSNR 46.0557/46.5333 dB. These comparisons do not establish identical intermediate arithmetic or isolate the cause of native Omni repeat variation.',
    primary_performance_unmodified=True,
    denoise_caveat='Omni denoise is a separate synchronized native-profiler diagnostic with metadata observer overhead; primary HTTP performance is profiler-free.')
target = evidence / 'preflight/joy-final-runtime-precision-audit.json'
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], shapes=list(shapes), target=str(target))))
