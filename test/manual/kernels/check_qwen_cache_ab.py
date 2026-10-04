"""Same-runner A/B diagnosis; reports existing test failures without changing gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
BASE = '5606fb8592d37e96fa4f795437958565d9c332df'
CANDIDATE = '715a644848607f5d3c526410c4af4d1370800081'
TASK = Path(tempfile.mkdtemp(prefix='qwen-cache-ab-'))
BASE_ROOT = TASK / 'baseline'
subprocess.run(['git','fetch','--depth=1','origin',BASE],cwd=ROOT,check=True)
subprocess.run(['git','worktree','add','--detach',str(BASE_ROOT),BASE],cwd=ROOT,check=True)
print('GPU metadata:',subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True),flush=True)
print('CPU:',subprocess.check_output(['lscpu'],text=True),flush=True)

child = r'''
import json, os, sys
import pytest
import torch
import sglang
from sglang.multimodal_gen.test.server.test_server_common import DiffusionServerBase
original = DiffusionServerBase._test_diffusion_request

def cold_request(self, *args, **kwargs):
    props = torch.cuda.get_device_properties(0)
    eviction = torch.zeros(props.L2_cache_size * 5, device='cuda', dtype=torch.uint8)
    eviction.add_(1)
    torch.cuda.synchronize()
    print('COLD_REQUEST',json.dumps({'l2_bytes':props.L2_cache_size,'eviction_bytes':eviction.numel(),'source':sglang.__file__}),flush=True)
    del eviction
    return original(self, *args, **kwargs)

DiffusionServerBase._test_diffusion_request = cold_request
sys.exit(pytest.main([sys.argv[1], '-k', 'qwen_image_t2i_cache_dit_enabled', '-x', '-s']))
'''
results=[]
try:
    for i, variant in enumerate(['A','B','B','A']*2):
        checkout = BASE_ROOT if variant=='A' else ROOT
        out = TASK / str(i)
        out.mkdir()
        env = dict(os.environ, PYTHONPATH=str(checkout / 'python') + os.pathsep + os.environ.get('PYTHONPATH',''))
        test = checkout / 'python/sglang/multimodal_gen/test/server/test_server_1_gpu.py'
        proc = subprocess.run([sys.executable,'-c',child,str(test)],cwd=out,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=900)
        (out/'test.log').write_text(proc.stdout)
        path = out/'diffusion-results.json'
        entry = dict(index=i,variant=variant,revision=BASE if variant=='A' else CANDIDATE,exit_code=proc.returncode)
        if path.exists():
            entry['metrics']=json.loads(path.read_text())
        else:
            entry['metrics']=[]
            print(proc.stdout[-5000:],flush=True)
        entry['cold_request']=[line for line in proc.stdout.splitlines() if line.startswith('COLD_REQUEST')]
        print('AB_RESULT',json.dumps(entry),flush=True)
        results.append(entry)
    (TASK/'results.json').write_text(json.dumps(results,indent=2))
    print('AB_REPORT',json.dumps(results),flush=True)
    assert all(r['metrics'] and r['cold_request'] for r in results), 'Incomplete measurement; inspect logs'
    print('Diagnostic completed; individual pytest exit codes above remain failures where applicable.',flush=True)
finally:
    subprocess.run(['git','worktree','remove',str(BASE_ROOT)],cwd=ROOT,check=False)
