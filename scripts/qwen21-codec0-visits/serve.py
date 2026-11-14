"""Own and stop only the recorded native server process group."""
import json
import os
import signal
import sys
from pathlib import Path

action, arm, run = sys.argv[1:]
assert arm in ('sgl','omni') and Path(run).name==run
folder=Path('/task/runs')/run
identity=folder/'server-identity.json'
if action=='start':
    assert not identity.exists()
    visit=json.loads((folder/'visit.json').read_text())
    command=(['/opt/sglang/bin/sglang','serve'] if arm=='sgl' else ['/task/envs/omni/bin/vllm','serve'])+visit['server_args'][arm]
    pid=os.getpid()
    if os.getsid(0)!=pid:os.setsid()
    assert os.getsid(0)==os.getpgrp()==pid
    os.environ.update(visit.get('server_env',{}).get(arm,{}))
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str, visit['physical_gpus']))
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['FLASHINFER_DISABLE_VERSION_CHECK']='1'
    os.environ['SGLANG_DIFFUSION_SYNC_STAGE_PROFILING']='1' if visit.get('synchronized_stage_profiling') else '0'
    if arm=='omni':
        os.environ['PATH']='/task/envs/omni/bin:'+os.environ['PATH']
        os.environ['LD_PRELOAD']='/task/envs/omni-cudnn/nvidia/cudnn/lib/libcudnn.so.9'
        os.environ['LD_LIBRARY_PATH']='/task/envs/omni-cudnn/nvidia/cudnn/lib:'+os.environ.get('LD_LIBRARY_PATH','')
        os.environ['VLLM_OMNI_SERVER_STORAGE__PATH']=str(folder/'server-storage')
    identity.write_text(json.dumps(dict(pid=pid,pgid=pid,
        start_ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19],command=command,
        explicit_environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','FLASHINFER_DISABLE_VERSION_CHECK','SGLANG_DIFFUSION_SYNC_STAGE_PROFILING','LD_PRELOAD','LD_LIBRARY_PATH','VLLM_OMNI_SERVER_STORAGE__PATH']}),indent=2)+'\n')
    os.execv(command[0],command)
else:
    assert action in ('term','kill')
    j=json.loads(identity.read_text());pid=j['pid'];p=Path(f'/proc/{pid}/stat')
    if p.exists():
        assert p.read_text().rsplit(')',1)[1].split()[19]==j['start_ticks'],'PID reused'
        assert os.getpgid(pid)==j['pgid']==pid
        os.killpg(pid,signal.SIGTERM if action=='term' else signal.SIGKILL)
