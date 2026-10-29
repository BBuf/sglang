"""Compare actual native runtime observations after output-preserving instrumentation."""
import hashlib
import json
from pathlib import Path

import torch

torch.set_num_threads(4)
root = Path('/task')
base = root / 'outputs/diagnostics/sana-video-tensor-audit1'
out = root / 'evidence/preflight/sana-video-runtime-input-audit1.json'
assert not out.exists()
status = json.loads((root / 'sana-video-tensor-audit1-status.json').read_text())
assert status['status'] == 'observed_native_outputs_exact_runtime_tensor_audit_pending'
assert len(status['comparisons']) == 8
assert all(row['observer_MP4_exact_native'] for row in status['comparisons'])


def read(arm, kind):
    records = []
    for path in (base / arm).glob('*/' + kind + '-*.pt'):
        record = torch.load(path, map_location='cpu', weights_only=True)
        record.update(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
    return sorted(records, key=lambda value: value['time_ns'])


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def metric(a, b):
    if a.shape != b.shape:
        return dict(exact=False, shape_a=list(a.shape), shape_b=list(b.shape))
    delta = a.float() - b.float()
    return dict(exact=torch.equal(a, b), shape=list(a.shape), dtype_a=str(a.dtype), dtype_b=str(b.dtype),
                max_abs=float(delta.abs().max()), relative_l2=float(delta.norm() / a.float().norm().clamp_min(1e-30)),
                sha256_a=digest(a), sha256_b=digest(b))


records = {}
for arm in ['sgl', 'omni']:
    records[arm] = dict(text=read(arm, arm + '-text-encoder'), steps=read(arm, 'scheduler-step'))
    assert len(records[arm]['text']) >= 8 and len(records[arm]['steps']) >= 8
cpu_noise = torch.randn((1, 16, 21, 60, 104), generator=torch.Generator(device='cpu').manual_seed(1101), dtype=torch.float32)
rows = []
for i, prompt in enumerate(['short', 'long', 'chinese', 'short-repeat']):
    conditions = []
    for j, branch in enumerate(['positive', 'negative']):
        a, b = [records[arm]['text'][-8:][2 * i + j] for arm in ['sgl', 'omni']]
        ai, bi = a['inputs'], b['inputs']
        window = [0] + list(range(-299, 0)) if branch == 'positive' else list(range(300))
        conditions.append(dict(branch=branch,
            token_ids=metric(ai['input_ids'], bi['input_ids']),
            attention_mask=metric(ai['attention_mask'], bi['attention_mask']),
            raw_hidden_states=metric(a['output'], b['output']),
            selected_hidden_states=metric(a['output'][:, window], b['output'][:, window]),
            selected_mask=metric(ai['attention_mask'][:, window], bi['attention_mask'][:, window]),
            source_files=[a['file'], b['file']]))
    a, b = [records[arm]['steps'][-8:][2 * i] for arm in ['sgl', 'omni']]
    af, bf = [records[arm]['steps'][-8:][2 * i + 1] for arm in ['sgl', 'omni']]
    assert a['step_index_before'] in [None, 0] and b['step_index_before'] in [None, 0]
    assert af['step_index_before'] == bf['step_index_before'] == 49
    rows.append(dict(prompt=prompt, conditions=conditions,
        initial_noise=metric(a['sample'], b['sample']),
        initial_noise_vs_cpu_seed1101=metric(a['sample'], cpu_noise),
        timesteps=metric(a['timesteps'], b['timesteps']), sigmas=metric(a['sigmas'], b['sigmas']),
        first_timestep=metric(a['timestep'], b['timestep']),
        first_prediction=metric(a['model_output'], b['model_output']),
        first_scheduler_output=metric(a['result'], b['result']),
        final_scheduler_output=metric(af['result'], bf['result']),
        source_files=[a['file'], b['file'], af['file'], bf['file']]))
result = dict(scope='Actual native SANA-Video runtime; observer returns untouched tensors. All8observed MP4s byte-exact original unobserved native outputs.',
    observed_outputs_byte_exact=True, requests=rows,
    noise_and_schedule_exact=all(all(row[key]['exact'] for key in ['initial_noise', 'initial_noise_vs_cpu_seed1101', 'timesteps', 'sigmas', 'first_timestep']) for row in rows),
    token_ids_masks_exact=all(all(c[key]['exact'] for key in ['token_ids', 'attention_mask', 'selected_mask']) for row in rows for c in row['conditions']),
    record_counts={arm: {kind: len(values) for kind, values in record.items()} for arm, record in records.items()},
    excluded_prefix_counts={arm: {kind: len(values) - 8 for kind, values in record.items()} for arm, record in records.items()},
    record_manifest=[dict(file=value['file'], sha256=value['sha256'], kind=value['kind'], index=value['index'], time_ns=value['time_ns']) for record in records.values() for values in record.values() for value in values],
    selected_window_note='Derived from recorded raw native hidden states and IDs/masks using both source implementations: positive BOS plus final299; negative300 unchanged. No injected conditioning or numerical override.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key not in ['requests', 'record_manifest']}))
for row in rows:
    print(row['prompt'], [(c['branch'], c['raw_hidden_states'].get('relative_l2')) for c in row['conditions']], row['first_prediction'].get('relative_l2'))
