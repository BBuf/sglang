"""Run diagnostics only after the identified primary process exits successfully."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
STATUS = ROOT/'joy-gelu-final-diagnostics1-queue-status.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(pid):
    folder = Path('/proc')/str(pid)
    try:
        fields = (folder/'stat').read_text().rsplit(')',1)[1].split()
        return dict(start_ticks=fields[19], command=(folder/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
    except FileNotFoundError:
        return None


def read(path):
    return json.loads(path.read_text())


def verify_files():
    manifest = read(ROOT/'joy-final-diagnostics-queue-manifest.json')
    for item in manifest['files']:
        assert hashlib.sha256((ROOT/item['remote_relative']).read_bytes()).hexdigest()==item['sha256'], item['remote_relative']
    return hashlib.sha256((ROOT/'joy-final-diagnostics-queue-manifest.json').read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-pid', type=int, required=True)
    args = parser.parse_args()
    assert not (ROOT/'joy-gelu-final-diagnostics1-status.json').exists()
    primary_identity = identity(args.primary_pid)
    assert primary_identity and primary_identity['command'].endswith('python3 run_joy_final_primary.py')
    manifest_sha = verify_files()
    # Exclusive creation prevents a second waiter from launching the same run.
    with STATUS.open('x') as stream:
        state = dict(status='waiting_for_primary_exit', queued_at=now(), pid=os.getpid(),
                     primary_pid=args.primary_pid, primary_identity=primary_identity,
                     manifest_sha256=manifest_sha, gpu_work_started=False)
        stream.write(json.dumps(state,indent=2)+'\n')

    def save():
        STATUS.write_text(json.dumps(state,indent=2)+'\n')

    try:
        while identity(args.primary_pid)==primary_identity:
            time.sleep(5)
        primary = read(ROOT/'joy-gelu-final-primary1-status.json')
        assert primary['status']=='raw_primary_complete_audit_and_final_diagnostics_pending', primary
        plans = [read(ROOT/name) for name in primary['protocol']['plans']]
        assert primary['completed']==[plan['id'] for plan in plans]
        for plan in plans:
            if 'main_endpoint_noise_check' in plan:
                screen = next(s for s in primary['screens'] if s['plan']==plan['id'])
                assert screen['main_endpoint_relative_difference']<=plan['main_endpoint_noise_check']['max_relative_difference'], 'Primary endpoint drift needs review'
        assert verify_files()==manifest_sha
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPUs not idle'
        assert not (ROOT/'joy-gelu-final-diagnostics1-status.json').exists()
        child = subprocess.Popen([sys.executable,str(ROOT/'run_joy_final_diagnostics.py')],cwd=ROOT)
        state.update(status='diagnostics_running', primary_exit_observed_at=now(),
                     child_pid=child.pid, gpu_work_started=True)
        save()
        code = child.wait()
        state.update(returncode=code, finished_at=now())
        assert code==0, f'Diagnostics exited {code}'
        terminal = read(ROOT/'joy-gelu-final-diagnostics1-status.json')
        assert terminal['status']=='raw_diagnostics_complete_final_archive_audit_pending', terminal
        state['status']='diagnostics_complete_archive_audit_pending'
    except Exception as error:
        state.update(status='failed_no_retry',error=repr(error),finished_at=now())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
