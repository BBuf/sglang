"""Observe the existing forensic process, then archive and audit once."""
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/data/xyzhang/diffusion-rtx6000-20261026'
NAME = 'qwen21-cleanup-1024-forensics1'
PYTHON = ROOT/'execution/joy-repair-analysis-venv/bin/python'
STATUS = ROOT/'evidence/preflight'/(NAME+'-completion1-status.json')
assert not STATUS.exists()
launch = json.loads((ROOT/'evidence/preflight'/(NAME+'-launch.json')).read_text())
assert launch['pid'] == 1208985 and launch['failed_primary_preserved']
state = dict(status='observing_existing_forensics', started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             local_controller_pid=os.getpid(), remote_parent_pid=launch['pid'], publication_started=False,
             primary_timing=False, failed_primary_remains_failed=True)


def save():
    state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    partial = STATUS.with_suffix('.tmp')
    partial.write_text(json.dumps(state, indent=2)+'\n')
    partial.replace(STATUS)


def ssh(code):
    return subprocess.check_output(['ssh', '-o', 'ConnectTimeout=10', 'SGL-RTX6000-01', 'python3 -'],
                                   input=code, text=True)


try:
    save()
    last = None
    while True:
        result = json.loads(ssh(f'''import json,subprocess
from pathlib import Path
root=Path({REMOTE!r});name={NAME!r}
proc=subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','args='],capture_output=True,text=True)
cmd=proc.stdout.strip();alive=bool(cmd) and not cmd.startswith('[')
if alive:assert 'run_qwen21_current_head_diagnostics1.py '+str(root/(name+'.json')) in cmd,'PID reuse'
status=json.loads((root/(name+'-status.json')).read_text())
print(json.dumps({{'parent_alive':alive,'status':status}}))
'''))
        state['observed'] = result
        save()
        encoded = json.dumps(result, sort_keys=True)
        if encoded != last:
            print(json.dumps(dict(at=state['updated_at'], **result)), flush=True)
            last = encoded
        assert result['status']['status'] != 'failed', 'Raw forensic run failed; retain all artifacts'
        if not result['parent_alive']:
            assert result['status']['status'] == 'raw_complete_audit_pending'
            assert len(result['status']['completed']) == 2
            break
        time.sleep(10)
    state['status'] = 'archiving_completed_forensics'
    save()
    script = 'archive_qwen21_cleanup_forensics1.py'
    sha = hashlib.sha256((ROOT/'scripts'/script).read_bytes()).hexdigest()
    assert sha == launch['installed_or_verified_sha256'][script]
    with (ROOT/'evidence/preflight'/(NAME+'-completion1.log')).open('x') as log:
        log.write(ssh(f'''import hashlib,subprocess
from pathlib import Path
root=Path({REMOTE!r});p=root/{script!r}
assert hashlib.sha256(p.read_bytes()).hexdigest()=={sha!r}
subprocess.run(['python3',str(p),{NAME!r}],check=True)
'''))
        log.flush()
        state['status'] = 'transferring_completed_forensics'
        save()
        for suffix in ['-archive-manifest.json', '-archive.tar.gz']:
            dest = ROOT/'evidence/checkpoints'/(NAME+suffix)
            partial = dest.with_name(dest.name+'.partial')
            assert not dest.exists() and not partial.exists()
            subprocess.run(['scp', '-q', 'SGL-RTX6000-01:'+REMOTE+'/'+NAME+suffix, str(partial)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            partial.rename(dest)
        state['status'] = 'auditing_completed_forensics'
        save()
        for script, args in [('extract_qwen21_current_head_archive1.py', [NAME]),
                             ('audit_qwen21_cleanup_forensics1.py', [])]:
            subprocess.run([str(PYTHON), str(ROOT/'scripts'/script), *args],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    state['status'] = 'complete_raw_audited_profile_interpretation_pending'
    state['audit'] = 'evidence/preflight/'+NAME+'-audit.json'
    state['audit_sha256'] = hashlib.sha256((ROOT/state['audit']).read_bytes()).hexdigest()
    save()
    print(json.dumps(state), flush=True)
except Exception as exc:
    state.update(status='stopped_preserve_all_outputs_no_automatic_retry', error=repr(exc))
    save()
    raise
