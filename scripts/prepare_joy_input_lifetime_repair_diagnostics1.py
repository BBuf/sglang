"""Prepare final-repair diagnostics; do not deploy or start GPU work."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
old='joy-current-head-';new='joy-input-lifetime-repair1-'
changes={
 'run_joy_current_head_diagnostics1.py':'run_joy_input_lifetime_repair_diagnostics1.py',
 'audit_joy_current_head_diagnostics1.py':'audit_joy_input_lifetime_repair_diagnostics1.py',
 'archive_joy_current_head_diagnostics1.py':'archive_joy_input_lifetime_repair_diagnostics1.py',
}
for src,dst in changes.items():
    text=(root/'scripts'/src).read_text().replace(old,new)
    if src.startswith('audit_'):
        start=text.index("test = read(EVIDENCE/'preflight/joy-input-lifetime-repair1-tests1-admission.json')")
        end=text.index('chains, profiles, quality_rows, stages =',start)
        text=text[:start]+'''test = read(EVIDENCE/'preflight/joy-input-lifetime-repair1-tests-admission.json')
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
''' + text[end:]
        text=text.replace('supplemental existing-PR publication remain separate gates.', 'publication decision remain separate gates. No new optimization slot is assigned.')
    if src.startswith('archive_'):
        text=text.replace(", root/'joy-input-lifetime-repair1-tests1-status.json'",'')
        text=text.replace("('outputs/diagnostics', 'joy-input-lifetime-repair1-tests1')", "('outputs/diagnostics', 'joy-input-lifetime-repair1-red-attempt2'),\n                      ('outputs/diagnostics', 'joy-input-lifetime-repair1-green-attempt1')")
    path=root/'scripts'/dst
    assert not path.exists();path.write_text(text);ast.parse(text)
plan=json.loads((root/'evidence/plans/joy-current-head-diagnostics1.json').read_text())
plan.update(id='joy-input-lifetime-repair1-diagnostics1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    purpose='Separate complete-model quality-switching and synchronized profiles for isolated lifetime repair 03368cf1; not evidence for unchanged official PR head b08c24e9.',
    no_new_optimization_PR_or_slot=True,official_PR_head_qualified=False)
plan['sources']['current'].update(repo='/task/candidates/joy-input-lifetime-repair1',head='03368cf1c325e526184b8fb223443e1ae931b13c')
plan['required_admissions']=[
 'evidence/preflight/joy-input-lifetime-repair1-tests-admission.json',
 'evidence/preflight/joy-input-lifetime-repair1-square-primary1-admission.json',
 'evidence/preflight/joy-input-lifetime-repair1-wide-primary1-admission.json']
plan['sha256'].pop('scripts/run_joy_current_head_tests1.py')
for src,dst in changes.items():
    plan['sha256'].pop('scripts/'+src)
    plan['sha256']['scripts/'+dst]=hashlib.sha256((root/'scripts'/dst).read_bytes()).hexdigest()
for relative in ['evidence/preflight/joy-current-head-tests1-admission.json','evidence/preflight/joy-input-lifetime-repair1-tests-admission.json']:
    plan['sha256'][relative]=hashlib.sha256((root/relative).read_bytes()).hexdigest()
# Pin exact actual red/green records in addition to the independent local receipt chain.
for phase in ['red-attempt2','green-attempt1']:
    folder=root/'evidence/diagnostics'/('joy-input-lifetime-repair1-'+phase)
    for name in ['receipt.json','junit.xml','pytest.log','preflight.log','source.patch']:
        plan['sha256']['outputs/diagnostics/'+folder.name+'/'+name]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
for relative,sha in plan['sha256'].items():
    p=root/relative
    if relative.startswith('outputs/'):
        p=root/'evidence'/Path(relative).relative_to('outputs')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,relative
path=root/'evidence/plans'/('joy-input-lifetime-repair1-diagnostics1.json');assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n')
receipt=dict(prepared_at=plan['created_at'],plan=str(path.relative_to(root)),plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    static_checked_scripts=list(changes.values()),payload_pins_verified=len(plan['sha256']),
    expected_actual_images=76,expected_actual_traces=4,GPU_executed=False,deployed=False,
    prerequisite='Both new repair primary admissions and local complete archive verifications. No failed official-head gate changed.')
path=root/'evidence/preflight/joy-input-lifetime-repair1-diagnostics-preparation.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
