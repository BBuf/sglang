"""Admit equal actual PNG encoding, then freeze and run the corrected comparison."""
import datetime
import hashlib
import json
import struct
import subprocess
import zlib
from pathlib import Path

root = Path('/data/xyzhang/diffusion-6000-20261113')
status = root / 'evidence/preflight/qwen21-codec0-status.json'
assert not status.exists()
heads = dict(main='0745bfe49b855ae48e0c1ef3305362b8e0c78a03',
             eager='0745bfe49b855ae48e0c1ef3305362b8e0c78a03',
             PR='15b28203ba81c96fac1b3692d7bc67bac01d8ff5',
             omni='aa2968e858a718989332a61ad40758024cb4a3a5')
state = dict(benchmark=False, visits=[], actual_png_compress_level=0)


def save(**values):
    state.update(values, at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status.write_text(json.dumps(state, indent=2) + '\n')


def execute(command, label):
    with (status.parent / (label + '-driver.log')).open('xb') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        save(status=label, active_pid=process.pid, command=command)
        code = process.wait()
    save(active_pid=None)
    assert code == 0, (label, code)


def visit(run, mode, plan=None, require_references=True):
    arm = 'omni' if mode == 'omni' else 'sgl'
    execute(['python3', str(root / 'scripts/qwen21-codec0-visits/run.py'), arm,
             run, plan or 'inputs/' + run + '.json'], run)
    result = json.loads((root / 'runs' / run / 'status.json').read_text())
    assert result['status'] == 'raw_complete' and result['gpu_cleanup'] == 'idle_verified'
    assert result['head_sha'] == heads[mode]
    if require_references:
        assert all(result['reference_checks'])
    state['visits'].append(dict(run=run, mode=mode, head_sha=result['head_sha']))
    save()


def png_stored_blocks(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    position, chunks = 8, []
    while position < len(data):
        size = struct.unpack('>I', data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + size]
        checksum = struct.unpack('>I', data[position + 8 + size:position + 12 + size])[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == checksum
        if kind == b'IDAT':
            chunks.append(payload)
        position += size + 12
    stream = b''.join(chunks)
    assert not (stream[1] & 32), 'unexpected zlib dictionary'
    raw = zlib.decompress(stream)
    position, blocks, stored_bytes = 2, 0, 0
    while True:
        header = stream[position]
        assert (header >> 1) & 3 == 0, 'PNG is not stored-block level0 output'
        size, complement = struct.unpack('<HH', stream[position + 1:position + 5])
        assert size ^ complement == 65535
        position += 5 + size
        stored_bytes += size
        blocks += 1
        if header & 1:
            break
    assert position == len(stream) - 4 and stored_bytes == len(raw)
    return dict(png_sha256=hashlib.sha256(data).hexdigest(), png_bytes=len(data),
                all_deflate_blocks_stored=True, stored_blocks=blocks,
                inflated_bytes=len(raw), idat_bytes=len(stream))


try:
    previous = json.loads((root / 'evidence/preflight/qwen21-prefix-release-admissions-validated.json').read_text())
    assert previous['status'] == 'release_admissions_verified_remaining_codec9_measurements_superseded'
    assert previous['head_sha'] == heads['PR']
    assert all(x['capture_fusion_calls'] == [32, 32, 32] for x in previous['high_capture_intervals'])
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    admitted, equivalence = {}, []
    for mode, old in [('main', 'qwen21-base-sgl-cudnn-bcg-01'),
                      ('PR', 'qwen21-prefix-release-cudnn-admission-01')]:
        run = f'qwen21-codec0-{mode}-admission-01'
        visit(run, mode, require_references=False)
        rows = json.loads((root / 'runs' / run / 'outputs/results.json').read_text())
        old_rows = json.loads((root / 'runs' / old / 'outputs/results.json').read_text())
        assert len(rows) == len(old_rows) == 6
        admitted[mode] = {row['prompt_id']: row['sha256'] for row in rows}
        for row, reference in zip(rows, old_rows, strict=True):
            assert row['decoded_rgba_sha256'] == reference['decoded_rgba_sha256']
            proof = png_stored_blocks(root / Path(row['output']).relative_to('/task'))
            equivalence.append(dict(mode=mode, index=row['index'], prompt_id=row['prompt_id'],
                before_png=reference['sha256'], after_png=row['sha256'],
                decoded_RGBA_exact=True, decoded_rgba_sha256=row['decoded_rgba_sha256'], **proof))
    omni_rows = json.loads((root / 'runs/qwen21-base-omni-01/outputs/results.json').read_text())
    for row in omni_rows:
        proof = png_stored_blocks(root / Path(row['output']).relative_to('/task'))
        equivalence.append(dict(mode='omni', index=row['index'], **proof))
    (status.parent / 'qwen21-codec0-actual-encoding-equivalence.json').write_text(json.dumps(equivalence, indent=2) + '\n')
    cpu = ['docker', 'exec', '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'OMP_NUM_THREADS=1',
           '-e', 'PYTHONPATH=/task/envs/quality-tools', 'bbuf-diffusion-6000-20261113-sgl', '/opt/sglang/bin/python']
    for name, left, right, labels in [
        ('qwen21-codec0-main-PR-01', 'qwen21-codec0-main-admission-01', 'qwen21-codec0-PR-admission-01', 'inputs/qwen21-codec0-main-PR-labels.json'),
        ('qwen21-codec0-cross-framework-01', 'qwen21-codec0-main-admission-01', 'qwen21-base-omni-01', 'inputs/qwen21-codec0-cross-labels.json')]:
        execute([*cpu, '/task/scripts/audit_qwen21_images.py', left, right, name, labels], name)
    assert json.loads((root / 'evidence/quality/qwen21-codec0-main-PR-01/summary.json').read_text())['numeric_passed']
    assert json.loads((root / 'evidence/quality/qwen21-codec0-cross-framework-01/cross-image-edit/metrics.json').read_text())['numeric_passed']
    plans = {}
    for mode in ('eager', 'main', 'PR', 'omni'):
        plan = json.loads((root / 'inputs' / f'qwen21-codec0-{mode}-quiet-template.json').read_text())
        if mode != 'omni':
            for request in plan['requests']:
                request['expected_sha256'] = {'sgl': admitted['main' if mode == 'eager' else mode][request['prompt_id']]}
        path = root / 'inputs' / f'qwen21-codec0-{mode}-quiet-01.json'
        assert not path.exists()
        path.write_text(json.dumps(plan, indent=2) + '\n')
        plans[mode] = str(path.relative_to(root))
    visits = []
    for phase, order in [('AA', ('eager', 'main', 'main', 'eager')),
                         ('paired', ('omni', 'main', 'PR', 'PR', 'main', 'omni'))]:
        counts = {}
        for mode in order:
            counts[mode] = counts.get(mode, 0) + 1
            visits.append(dict(mode=mode, phase=phase, arm='omni' if mode == 'omni' else 'sgl',
                run=f'qwen21-codec0-{mode}-{phase}-{counts[mode]:02d}', plan=plans[mode]))
    rule = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), heads=heads,
                visits=visits, physical_gpus=[0], actual_png_compress_level=0,
                API_values={'sglang': 0, 'omni': 100},
                prior_non_equivalent_E2E_retained_but_excluded=True,
                plan_sha256={path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in plans.values()},
                protocol='E-M-M-E AA, then O-M-F-F-M-O paired; each2warm+7measure; full50step1024edit. E=main eager,M=mainBCG,F=PR15bBCG. All requests admitted before timing; allactualPNGdeflateblocks stored, decodedRGBA unchanged frompreviousencoding. No source change, SGLcompile/offload or disabledOmnifastpath.')
    rule_path = root / 'inputs/qwen21-codec0-quiet-rule-01.json'
    assert not rule_path.exists()
    rule_path.write_text(json.dumps(rule, indent=2) + '\n')
    save(benchmark=True, rule_sha256=hashlib.sha256(rule_path.read_bytes()).hexdigest())
    for v in visits:
        assert hashlib.sha256((root / v['plan']).read_bytes()).hexdigest() == rule['plan_sha256'][v['plan']]
        visit(v['run'], v['mode'], v['plan'])
    for index, mode in enumerate(('main', 'PR', 'PR', 'main')):
        plan = json.loads((root / plans[mode]).read_text())
        plan['requests'] = plan['requests'][:5]
        plan['synchronized_stage_profiling'] = True
        plan['purpose'] = 'same_codec0_separate_native_synchronized_stages'
        run = f'qwen21-codec0-{mode}-stages-{1 if index < 2 else 2:02d}'
        path = root / 'inputs' / (run + '.json')
        assert not path.exists()
        path.write_text(json.dumps(plan, indent=2) + '\n')
        visit(run, mode)
    save(benchmark=False)
    for mode in ('main', 'PR'):
        run = f'qwen21-codec0-{mode}-profile-01'
        plan = json.loads((root / 'inputs' / f'qwen21-codec0-{mode}-profile-template.json').read_text())
        for request in plan['requests']:
            request['expected_sha256'] = {'sgl': admitted[mode][request['prompt_id']]}
        path = root / 'inputs' / (run + '.json')
        assert not path.exists()
        path.write_text(json.dumps(plan, indent=2) + '\n')
        visit(run, mode)
        execute([*cpu, '/task/scripts/analyze_qwen21_trace_costs.py', run], run + '-costs')
    save(status='codec0_all_measurements_complete_analysis_pending')
except BaseException as exc:
    save(status='failed_preserve_no_retry', error=repr(exc))
    raise
