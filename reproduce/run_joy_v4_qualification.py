"""Qualify all declared Joy v4 shapes/modes serially, preserving every output."""
import hashlib
import json
import subprocess
from pathlib import Path

import run_joy_candidate_qualification as harness

ROOT = harness.ROOT
plan = json.loads((ROOT / 'joy-gelu-v4-qualification1-plan.json').read_text())
assert json.loads((ROOT / 'joy-native-aspect-input-comparison.json').read_text())['all_pass']
for contract in plan['contracts'].values():
    path = ROOT / Path(contract['path']).relative_to('/task')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == contract['sha256']
weights = json.loads((ROOT / 'evidence/downloads/joy-modelscope-download-manifest.json').read_text())
assert weights['complete'] and len(weights['files']) == 36 and all(f['verified_against_hf'] for f in weights['files'])
assert weights['hf_revision'] == '4b41fb25d961f37668750178ccbb380da326201c'
status_path = ROOT / 'joy-gelu-v4-qualification1-status.json'
assert not status_path.exists()
status = dict(status='running', plan=plan, completed=[], checks=[])


def check(candidate, references, identical=False):
    command = ['docker', 'exec', '-e', 'OMP_NUM_THREADS=1', 'bbuf-diffusion-rtx6000-20261026-omni',
               'python', '/task/compare_moss_qualification.py', candidate,
               '--root', '/task/outputs/qualification', '--fail-on-quality']
    for reference in references:
        command += ['--reference', reference]
    if identical:
        command += ['--require-pixel-identical']
    log = ROOT / (candidate+'-gate.log')
    with log.open('xb') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    status['checks'].append(dict(candidate=candidate, references=references, pixel_identical_required=identical, passed=True))


try:
    for stage in plan['order']:
        source = stage['source']
        status['active_source'] = source
        status_path.write_text(json.dumps(status, indent=2)+'\n')
        harness.run(source, plan['label'], stage['qualities'],
                    [Path(plan['contracts'][shape]['path']) for shape in stage['shapes']],
                    plan['candidate_head_sha'], plan['candidate_diff_sha256'], stage.get('regional_compile', False))
        if source == 'main':
            for shape in stage['shapes']:
                check(f'sgl-joy-gelu-v4-{shape}-main-lossless', [f'sgl-joy-gelu-v4-{shape}-main-high'], True)
        elif source == 'pr':
            for shape in stage['shapes']:
                prefix = 'sgl-joy-gelu-v1-main' if shape == 'square' else f'sgl-joy-gelu-v4-{shape}-main'
                references = [prefix+'-high']
                if shape == 'square':
                    references.append('omni-joy-compile-probe1-omni_compile-high')
                check(f'sgl-joy-gelu-v4-{shape}-pr-high', references)
                check(f'sgl-joy-gelu-v4-{shape}-pr-lossless', [prefix+'-lossless'], True)
        else:
            for shape in stage['shapes']:
                check(f'omni-joy-gelu-v4-{shape}-omni-high',
                      [f'sgl-joy-gelu-v4-{shape}-main-high', f'sgl-joy-gelu-v4-{shape}-pr-high'])
        status['completed'].append(source)
        print(json.dumps(dict(source=source, status='outputs_and_numeric_gates_passed')), flush=True)
    status.update(status='numeric_quality_passed_visual_review_pending', active_source=None)
except Exception as error:
    status.update(status='failed', error=repr(error))
    raise
finally:
    status_path.write_text(json.dumps(status, indent=2)+'\n')
