"""Start one explicit forensic run after verified failure; never retry primary."""
import argparse
import ast
import base64
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/data/xyzhang/diffusion-rtx6000-20261026'
CONTAINER = 'bbuf-diffusion-rtx6000-20261026-sgl'
NAME = 'qwen21-cleanup-1024-forensics1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--start', action='store_true')
args = parser.parse_args()
plan_path = ROOT/'evidence/plans'/(NAME+'.json')
plan = read(plan_path)
assert plan['id'] == NAME and not plan['primary_timing']
assert plan['order'] == ['original', 'current']
assert plan['expected_processes'] == 2 and plan['expected_outputs'] == 44
failure = read(ROOT/plan['preserved_failure']['admission'])
assert failure['passed'] is False
assert all(c['primary_passed'] and not c['cleanup_passed'] for c in failure['cases'].values())
verification = read(ROOT/plan['preserved_failure']['verification'])
assert verification['all_member_hashes_verified'] and verification['all_members_extracted']
assert verification['sha256'] == '125a94c6168aba7dbebcffe27d31ab19a0e1944b88cee161c10835034ee6ecdc'
followup = read(ROOT/'evidence/preflight/qwen21-current-head-after-primary1-status.json')
assert followup['status'] == 'stopped_preserve_all_outputs_no_automatic_retry'
assert followup['completed'] == [] and not followup['publication_started']
for pid, marker in [(75528, 'finish_qwen21_current_head_phase1.py'),
                    (78185, 'continue_qwen21_current_head_diagnostics1.py')]:
    proc = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    assert marker not in proc.stdout, 'Old controller still active'
payload = {NAME+'.json': plan_path.read_bytes()}
for name, expected in plan['sha256'].items():
    path = ROOT/name
    assert digest(path) == expected, name
    payload[Path(name).name if name.startswith('scripts/') else name] = path.read_bytes()
archive_name = 'archive_qwen21_cleanup_forensics1.py'
payload[archive_name] = (ROOT/'scripts'/archive_name).read_bytes()
pins = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
encoded = {name: base64.b64encode(data).decode() for name, data in payload.items()}
install = '''import base64,json
from pathlib import Path
payload=json.loads(%r)
for name,value in payload.items():
 p=Path('/task')/name;data=base64.b64decode(value,validate=True)
 assert not p.exists() or p.read_bytes()==data,name
for name,value in payload.items():
 p=Path('/task')/name;data=base64.b64decode(value,validate=True)
 if not p.exists():
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
 assert p.read_bytes()==data,name
''' % json.dumps(encoded)
remote = f'''import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path({REMOTE!r});name={NAME!r};pins={pins!r}
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
assert '0, GPU-1565b45e-a702-2015-9776-61f68c36fd59' in gpu
old=root/'qwen21-current-head-1024-primary1-status.json'
state=json.loads(old.read_text());assert state['status']=='raw_complete_audit_pending' and len(state['completed'])==14
for previous in {plan['not_run_original_plans']!r}:assert not (root/(previous+'-status.json')).exists()
for suffix in ['-status.json','-launch.json','-archive.tar.gz','-archive-manifest.json','-controller.log']:
 assert not (root/(name+suffix)).exists(),suffix
assert not list((root/'block-runs').glob(name+'-*'))
subprocess.run(['docker','exec','-i',{CONTAINER!r},'python','-'],input={install!r},text=True,check=True)
for path,sha in pins.items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==sha,path
plan=json.loads((root/(name+'.json')).read_text())
for path,sha in plan['reference_files_sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==sha,path
for path in plan['required_admissions']:assert json.loads((root/path).read_text())['passed'],path
failure=json.loads((root/plan['preserved_failure']['admission']).read_text());assert failure['passed'] is False
for spec in plan['configurations'].values():
 assert subprocess.check_output(['docker','exec',{CONTAINER!r},'git','-C',spec['repo'],'rev-parse','HEAD'],text=True).strip()==spec['head']
 assert not subprocess.check_output(['docker','exec',{CONTAINER!r},'git','-C',spec['repo'],'status','--porcelain'])
cmd=['python3','-u',str(root/'run_qwen21_current_head_diagnostics1.py'),str(root/(name+'.json'))]
with (root/(name+'-controller.log')).open('xb') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':name,'pid':proc.pid,'command':cmd,'primary_timing':False,'failed_primary_preserved':True,'plan_sha256':pins[name+'.json'],'installed_or_verified_sha256':pins,'reference_files_verified':len(plan['reference_files_sha256'])}}
with (root/(name+'-launch.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\\n')
print(json.dumps(receipt))
'''
ast.parse(remote)
ast.parse(install)
for path in [ROOT/'scripts/run_qwen21_current_head_diagnostics1.py', ROOT/'scripts'/archive_name]:
    ast.parse(path.read_text())
if not args.start:
    print(json.dumps(dict(checked=True, started=False, plan_sha256=digest(plan_path), files=len(pins))))
else:
    output = subprocess.check_output(['ssh', '-o', 'ConnectTimeout=10', 'SGL-RTX6000-01', 'python3 -'],
                                     input=remote, text=True)
    receipt = json.loads(output)
    receipt['local_launcher_sha256'] = digest(Path(__file__))
    with (ROOT/'evidence/preflight'/(NAME+'-launch.json')).open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))
