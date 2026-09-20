"""Four fixed native CLI diagnostics; sampling/GC logs are not speed evidence."""
import os
from pathlib import Path
import subprocess
import time
root=Path('/campaign')
done=root/'longlive2-norm-audit-retry3.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0',done
out=root/'artifacts/longlive2/client-diagnostic-retry2';out.mkdir(exist_ok=False)
shim=out/'bin';shim.mkdir()
entry=out/'entry.py'
entry.write_text('''import gc, json, os, runpy, sys, time
from pathlib import Path
log=Path(os.environ["LONGLIVE_CLIENT_DIAG"])/("gc-"+str(os.getpid())+".jsonl")
stream=log.open("a",buffering=1)
starts={}
def callback(phase, info):
    now=time.perf_counter(); gen=info["generation"]
    if phase=="start":starts[gen]=now
    elif gen in starts:
        dt=now-starts.pop(gen)
        if dt>0.01:stream.write(json.dumps(dict(pid=os.getpid(),wall=time.time(),duration_s=dt,**info))+"\\n")
gc.callbacks.append(callback)
sys.argv[0]="/opt/sglang/bin/sglang"
runpy.run_path(sys.argv[0],run_name="__main__")
''')
exe=shim/'sglang'
exe.write_text('#!/bin/bash\n/opt/sglang/bin/python /campaign/artifacts/longlive2/client-diagnostic-retry2/entry.py "$@" &\ndiag_child=$!\n/opt/sglang/bin/py-spy record --pid "$diag_child" --rate 100 --format speedscope --output "$LONGLIVE_CLIENT_DIAG/samples.json"\nspy_status=$?\nwait "$diag_child"\nchild_status=$?\nprintf "%s\\n" "$spy_status" > "$LONGLIVE_CLIENT_DIAG/sampler.exit"\nexit "$child_status"\n');exe.chmod(0o755)
for arm in ('a1','b1','b2','a2'):
    label='diag-t2v-client-retry2-'+arm;cell=out/arm;cell.mkdir()
    env=os.environ|dict(PATH=str(shim)+':'+os.environ['PATH'],LONGLIVE_CLIENT_DIAG=str(cell))
    repo=root/('baseline' if arm.startswith('a') else 'candidate-longlive2-wan-norm')
    with (root/('longlive2-'+label+'.log')).open('x') as log:
        p=subprocess.run(['python','-u',str(root/'run-longlive2.py'),'--repo',str(repo),'--mode','t2v','--label',label],env=env,stdout=log,stderr=subprocess.STDOUT)
    (cell/'exit').write_text(str(p.returncode));assert p.returncode==0,label
print('Four diagnostics completed; exclude their timings from performance claims.')
