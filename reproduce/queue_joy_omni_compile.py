"""Continue independent native Omni qualification after owned SGL quality jobs stop."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/data/xyzhang/diffusion-rtx6000-20261026')
meta=ROOT/'block-runs/joy-gelu-v1-pr/metadata.json'
start=time.monotonic()
while True:
    ready=False
    if meta.exists():
        try:
            ready=json.loads(meta.read_text()).get('status')=='quality_outputs_generated_not_yet_compared'
        except json.JSONDecodeError:
            pass
    idle=not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if ready and idle:
        break
    assert time.monotonic()-start<1800,'Preceding owned quality run not complete; no new GPU launch'
    time.sleep(5)
rows=[]
for arm in ['omni_compile','omni_compile_casts']:
    args=['python3',str(ROOT/'run_joy_omni_compile_qualification.py'),'--label','compile-probe1','--arms',arm,'--qualities','high']
    result=subprocess.run(args,cwd=ROOT)
    rows.append(dict(arm=arm,command=args,exit_code=result.returncode))
    (ROOT/'joy-compile-probe1-driver-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU processes remain; stop'
print(json.dumps(rows),flush=True)
