"""Strict serial final-run queue; stop on any failure, never retry or overwrite."""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/data/xyzhang/diffusion-rtx6000-20261026')
NAME='qwen21-final-validation-queue1'
STATUS=ROOT/(NAME+'-status.json')
assert not STATUS.exists()
items=[
 ('qwen21-gate-up-v4-final-512-compare1','run_qwen21_blocks_v4.py'),
 ('qwen21-gate-up-v4-final-1024-compare1','run_qwen21_blocks_v4.py'),
 ('qwen21-gate-up-v4-final-512-diagnostics1','run_qwen21_final_diagnostics.py'),
 ('qwen21-gate-up-v4-final-1024-diagnostics1','run_qwen21_final_diagnostics.py'),
]
previous='qwen21-cleanup-within-1024-1'
state=dict(status='waiting',completed=[],plan=items,waiting_for=previous)
def save():
 state['updated_at']=datetime.now(timezone.utc).isoformat();STATUS.write_text(json.dumps(state,indent=2)+'\n')
def terminal(run):
 path=ROOT/(run+'-status.json')
 while True:
  if path.exists():
   value=json.loads(path.read_text())
   if value['status']=='raw_complete_audit_pending':return
   if value['status'].startswith('failed'):raise RuntimeError(f'{run} failed: {value}')
  time.sleep(5)
try:
 for run,runner in items:
  state.update(status='waiting',waiting_for=previous,active=None);save();terminal(previous)
  command=['python3',str(ROOT/'launch_qwen21_plan.py'),run,'--requires',previous,'--runner',runner]
  output=subprocess.check_output(command,text=True);state.update(status='running',waiting_for=None,active=run,launch=json.loads(output));save()
  terminal(run);state['completed'].append(run);previous=run;save()
 state.update(status='complete_raw_audits_pending',active=None);save()
except Exception as error:
 state.update(status='failed',error=repr(error));save();raise
