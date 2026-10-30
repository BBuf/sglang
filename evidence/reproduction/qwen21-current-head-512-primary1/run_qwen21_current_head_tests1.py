"""Verify the pinned checkpoint and actual SM120 packing/lifecycle regression before timings."""
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path('/task')
head = '188169fa0db0bb2d524d56a97184590f948cca07'
repo = root/'candidates/qwen21-gate-up-pr47090-current'
run = 'qwen21-current-head-tests1'
status_path = root/(run+'-status.json')
assert not status_path.exists(), 'No automatic retry or overwrite'
archive = json.loads((root/'evidence/checkpoints/joy-input-lifetime-repair1-diagnostics1-local-verification.json').read_text())
assert archive['all_member_hashes_verified'] and archive['all_members_extracted']


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert idle()
assert command(['git', '-C', str(repo), 'rev-parse', 'HEAD']) == head
assert not command(['git', '-C', str(repo), 'diff', 'HEAD', '--'])
assert command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()[0] == 'GPU-1565b45e-a702-2015-9776-61f68c36fd59'
folder = root/'outputs/diagnostics'/run
folder.mkdir(exist_ok=False)
state = dict(status='checking_checkpoint', source_sha=head, verified_files=[])


def save():
    status_path.write_text(json.dumps(state, indent=2)+'\n')


save()
try:
    manifest_path = root/'evidence/downloads/qwen21-modelscope-download-manifest.json'
    assert digest(manifest_path) == 'cb4d933a7010de24ddfe75d99d70cc43a41b6459d9e25035ddcd7f5354e4f436'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and manifest['hf_revision'] == 'd26bb61231c349cf6b7896fa83353113880e1ba3'
    restored_path = root/'evidence/downloads/qwen21-current-head-restore1-modelscope-download-manifest.json'
    restored = json.loads(restored_path.read_text())
    assert restored['complete'] and restored['hf_revision'] == manifest['hf_revision']
    assert {row['path']:row['sha256'] for row in restored['files']} == {row['path']:row['sha256'] for row in manifest['files']}
    restore_plan = root/'evidence/plans/qwen21-current-head-restore1.json'
    assert restored['plan_sha256'] == digest(restore_plan)
    checkpoint = root/'cache/common/Qwen-Image-2.1'
    assert not checkpoint.is_symlink() and len(manifest['files']) == 24
    for row in manifest['files']:
        path = checkpoint/row['path']
        assert path.resolve().is_relative_to(checkpoint.resolve()) and not path.is_symlink()
        assert row['verified_against_hf'] and path.stat().st_size == row['bytes']
        assert digest(path) == row['sha256'], row['path']
        state['verified_files'].append(dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256']))
        save()
        print(json.dumps(dict(verified=row['path'], bytes=row['bytes'])), flush=True)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(repo/'python'), FLASHINFER_DISABLE_VERSION_CHECK='1',
                       HF_HUB_OFFLINE='1', OMP_NUM_THREADS='8')
    with (folder/'preflight.log').open('xb') as stream:
        subprocess.run([sys.executable, str(root/'preflight_diffusion_current_head1.py'), str(repo)],
                       env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    environment['CUDA_VISIBLE_DEVICES'] = '0'
    junit = folder/'junit.xml'
    test = 'test/registered/kernels/ops/diffusion/test_model_fast_paths.py::test_qwen21_bf16_mlp_packing_restore_and_lora'
    args = [sys.executable, '-m', 'pytest', '-q', test, '--junitxml='+str(junit)]
    state.update(status='running_regression', command=args)
    save()
    start = time.monotonic()
    with (folder/'pytest.log').open('xb') as stream:
        subprocess.run(args, cwd=repo, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=1800)
    seconds = time.monotonic()-start
    cases = list(ET.parse(junit).getroot().iter('testcase'))
    assert len(cases) == 2 and {case.attrib['name'] for case in cases} == {test.split('::')[-1]+'[1024]', test.split('::')[-1]+'[4096]'}
    assert all(case.find(name) is None for case in cases for name in ['failure', 'error', 'skipped'])
    for _ in range(30):
        if idle():
            break
        time.sleep(1)
    assert idle(), 'GPU processes remain after the owned regression'
    result = dict(passed=True, source_sha=head, tests_passed=2, tests_skipped=0,
                  test=test, command=args, seconds=seconds,
                  checkpoint_revision=manifest['hf_revision'], checkpoint_files_rehashed=len(state['verified_files']),
                  checkpoint_manifest_sha256=digest(manifest_path),
                  restoration_manifest_sha256=digest(restored_path),
                  files_sha256={str(p.relative_to(root)):digest(p) for p in folder.iterdir() if p.is_file()},
                  script_sha256=digest(Path(__file__)),
                  scope='Actual SM120 packing at1024/4096tokens, checkpoint names, restoration, prefix fallback, compile frontend, CPU/GPU repacking, detached storage and dynamic LoRA plus checkpoint provenance. Model performance, native quality/media and final profiles remain separate gates.')
    out = root/'evidence/preflight/qwen21-current-head-tests1-admission.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    state.update(status='passed', admission=str(out))
except Exception as exc:
    state.update(status='failed_preserve_outputs_no_retry', error=repr(exc))
    raise
finally:
    save()
