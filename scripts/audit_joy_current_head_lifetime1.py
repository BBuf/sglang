"""Verify archived lifetime observations and separate dispatch traces."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = 'joy-current-head-lifetime-probe1'
folder = root / 'evidence/diagnostics' / run


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


verification = read(root / 'evidence/checkpoints' / (run + '-local-verification.json'))
assert verification['all_member_hashes_verified'] and verification['all_members_extracted']
result = read(folder / 'results.json')
plan_path = root / 'evidence/plans' / (run + '.json')
plan = read(plan_path)
assert result['plan'] == plan and result['status'] == 'complete_causal_review_pending'
assert not result['full_model_admission'] and not result['timing_admission']
assert len(result['records']) == plan['expected_records'] == 8
assert digest(root / 'scripts/probe_joy_current_head_lifetime1.py') == plan['script_sha256']
assert digest(root / 'evidence/reproduction' / run / (run + '.json')) == digest(plan_path)
assert digest(root / 'evidence/reproduction' / run / (run + '.py')) == plan['script_sha256']
cases, trace_hashes = {}, {}
for tokens in plan['tokens']:
    rows = [r for r in result['records'] if r['tokens'] == tokens]
    assert [r['source'] for r in rows] == plan['order']
    assert [r['index'] for r in rows] == list(range(4))
    assert len({r['output_sha256'] for r in rows}) == 1
    assert len({r['before_allocated_bytes'] for r in rows}) == 1
    kernels = []
    for row in rows:
        assert row['output_exact'] and row['input_dead_after_return']
        assert row['lifetime_and_memory_observation_has_no_torch_profiler']
        entry = row['output_projection_entry']
        assert entry['input_still_alive'] == (row['source'] == 'current')
        assert entry['hidden_shape'] == [1, tokens, 16384]
        trace = root / 'evidence' / Path(row['trace']).relative_to('outputs')
        assert digest(trace) == row['trace_sha256']
        trace_hashes[str(trace.relative_to(root))] = row['trace_sha256']
        events = json.loads(gzip.decompress(trace.read_bytes()))['traceEvents']
        assert sum(e.get('cat') == 'user_annotation' and e.get('name') == row['source'] + '_real_MLP_temporary_input' for e in events) == 1
        assert not any(e.get('name') == 'cudaGraphLaunch' for e in events)
        counter = collections.Counter(e['name'] for e in events if e.get('cat') == 'kernel')
        assert sum(counter.values()) >= 2
        kernels.append(counter)
    original, current = rows[0], rows[1]
    delta = current['peak_allocated_bytes'] - original['peak_allocated_bytes']
    assert delta == tokens * 4096 * 2
    for row in rows:
        reference = current if row['source'] == 'current' else original
        assert row['peak_allocated_bytes'] == reference['peak_allocated_bytes']
        assert row['output_projection_entry']['allocated_bytes'] == reference['output_projection_entry']['allocated_bytes']
    cases[str(tokens)] = dict(extra_live_input_bytes=delta, extra_live_input_MiB=delta / 1024**2,
        original_peak_increment_bytes=original['peak_increment_bytes'],
        current_peak_increment_bytes=current['peak_increment_bytes'],
        identical_output_sha256=original['output_sha256'],
        same_profile_kernel_multiset=all(c == kernels[0] for c in kernels),
        profile_kernel_counts=[sum(c.values()) for c in kernels],
        profile_kernel_multisets=[dict(c) for c in kernels])
out = root / 'evidence/preflight' / (run + '-audit.json')
assert not out.exists()
value = dict(archive_integrity_passed=True, actual_lifetime_difference_confirmed=True,
    cases=cases, trace_sha256=trace_hashes, results_sha256=digest(folder / 'results.json'),
    plan_sha256=digest(plan_path), script_sha256=digest(Path(__file__)),
    full_model_admission=False, timing_causality_established=False,
    limitation='Unprofiled real MLP calls establish input retention and allocated-memory differences; separate traces inspect dispatch. This does not attribute all138MiB of model NVML peak or the0.0467% model timing regression, does not erase the failed cleanup gate, and is not a new PR qualification.')
out.write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps({k: {field: row[field] for field in ('extra_live_input_MiB', 'same_profile_kernel_multiset', 'profile_kernel_counts')}
                  for k, row in cases.items()}, indent=2))
