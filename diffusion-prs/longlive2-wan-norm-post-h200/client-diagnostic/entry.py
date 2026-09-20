import gc, json, os, runpy, sys, time
from pathlib import Path
log=Path(os.environ["LONGLIVE_CLIENT_DIAG"])/("gc-"+str(os.getpid())+".jsonl")
stream=log.open("a",buffering=1)
starts={}
def callback(phase, info):
    now=time.perf_counter(); gen=info["generation"]
    if phase=="start":starts[gen]=now
    elif gen in starts:
        dt=now-starts.pop(gen)
        if dt>0.01:stream.write(json.dumps(dict(pid=os.getpid(),wall=time.time(),duration_s=dt,**info))+"\n")
gc.callbacks.append(callback)
sys.argv[0]="/opt/sglang/bin/sglang"
runpy.run_path(sys.argv[0],run_name="__main__")
