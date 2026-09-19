"""Native FastH3 admission/profile on the campaign's two H200 GPUs."""
import argparse
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--repo',default='/campaign/baseline')
p.add_argument('--label',required=True)
p.add_argument('--profile',action='store_true')
p.add_argument('--bcg',action='store_true')
p.add_argument('--precision',choices=['bf16','fp8'],default='fp8')
p.add_argument('--duration',type=float,default=10.0)
args=p.parse_args()
root=Path('/campaign')
lock=(root/'gpu.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
repo=Path(args.repo).resolve()
os.environ.update(json.loads((root/'artifacts/fasth3-cache-env.json').read_text()))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(repo/'python'),OMP_NUM_THREADS='8')
sys.path.insert(0,str(repo/'python'))
os.chdir(repo)
import sglang
import torch
assert Path(sglang.__file__).resolve().is_relative_to(repo)
assert torch.cuda.device_count()==2
helper_dir=repo/'python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts'
subprocess.run(['python',str(helper_dir/'diffusion_skill_env.py'),'check-write-access'],check=True)
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
assert not any(line.split(',')[0].strip() in selected for line in processes.splitlines()),processes
subprocess.run(['git','diff','--quiet'],check=True)
subprocess.run(['git','diff','--cached','--quiet'],check=True)
out=root/'artifacts/fasth3'/args.label
out.mkdir(parents=True,exist_ok=False)
(out/'source.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True)+str(sglang.__file__)+'\n')
(out/'gpu-before.txt').write_text(inventory+'\nProcesses:\n'+processes)
spec=importlib.util.spec_from_file_location('campaign_bench',helper_dir/'bench_diffusion_denoise.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
preset=bench.MODELS['fasth3-t2va-vsa']
preset['config_overrides']['target']['duration_seconds']=args.duration
preset['extra_args']=['--num-gpus=2','--ulysses-degree=2','--ring-degree=1',
    '--component-attention-backends=transformer=video_sparse_attn_h3','--attention-backend-config={"VSA_sparsity":0.9}',
    '--enable-torch-compile=false','--performance-mode=speed','--warmup-steps=2',
    '--warmup-resolutions=1344x768',f'--warmup-num-frames={round(args.duration*24)}']
if args.precision=='fp8':
    preset['extra_args'].append('--quantization=fp8')
if args.profile:
    preset['extra_args']+=['--profile','--num-profiled-timesteps=2']
    os.environ['SGLANG_DIFFUSION_TORCH_PROFILER_DIR']=str(out/'traces')
(out/'preset.json').write_text(json.dumps(preset,indent=2))
result=bench._run_benchmark_once_impl('fasth3-t2va-vsa',args.label,out,warmup=True,torch_compile=False,
    quality='lossless',breakable_cuda_graph=args.bcg,model_cache_dir=root/'model-caches/fasth3-cycle',cuda_visible_devices='0,1')
if args.profile and not result.get('error'):
    traces=sorted((out/'traces').rglob('*.trace.json.gz'))
    if not traces:
        result.update(error=True,error_reason='Profile requested but no native trace was produced')
    else:
        evidence=[]
        for trace in traces:
            with gzip.open(trace,'rt') as stream:
                events=json.load(stream)['traceEvents']
            kernels=sum(e.get('cat')=='kernel' for e in events)
            assert kernels>0,trace
            evidence.append(dict(path=str(trace),event_count=len(events),kernel_count=kernels))
            del events
        result['native_profile_traces']=evidence
(out/'result.json').write_text(json.dumps(result,indent=2))
raise SystemExit(int(bool(result.get('error'))))
