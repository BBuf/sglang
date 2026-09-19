"""Pinned native LongCat Image Edit request with saved output and perf evidence."""
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
args=p.parse_args()
root=Path('/campaign')
lock=(root/'gpu.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
repo=Path(args.repo).resolve()
os.environ.update(json.loads((root/'artifacts/longcat-edit-cache-env.json').read_text()))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(repo/'python'),OMP_NUM_THREADS='8')
sys.path.insert(0,str(repo/'python'))
os.chdir(repo)
import sglang
import torch
assert Path(sglang.__file__).resolve().is_relative_to(repo)
assert torch.cuda.device_count()==1
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip()=='0'}
assert not any(line.split(',')[0].strip() in selected for line in processes.splitlines()),processes
subprocess.run(['git','diff','--quiet'],check=True)
subprocess.run(['git','diff','--cached','--quiet'],check=True)
out=root/'artifacts/longcat-image-edit'/args.label
out.mkdir(parents=True,exist_ok=False)
(out/'source.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True)+str(sglang.__file__)+'\n')
(out/'gpu-before.txt').write_text(inventory+'\nProcesses:\n'+processes)
helper=repo/'python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper)
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
preset=bench.MODELS['longcat-image-edit']
preset['seed']=42
preset['image_path']=str(root/'artifacts/input-media/longcat-edit-input.jpg')
preset['extra_args'] += [
    '--num-gpus=1', '--enable-torch-compile=false',
    '--num-inference-steps=50', '--guidance-scale=4.5',
    '--warmup-resolutions=1264x848',
]
if args.profile:
    preset['extra_args']+=['--profile','--num-profiled-timesteps=3']
    os.environ['SGLANG_DIFFUSION_TORCH_PROFILER_DIR']=str(out/'traces')
bench.MODELS['longcat-image-edit']=preset
(out/'preset.json').write_text(json.dumps(preset,indent=2))
result=bench._run_benchmark_once_impl('longcat-image-edit',args.label,out,warmup=True,torch_compile=False,
    quality='lossless',breakable_cuda_graph=args.bcg,model_cache_dir=root/'model-caches/longcat-image-edit-cycle',cuda_visible_devices='0')
if args.profile and not result.get('error'):
    traces=sorted((out/'traces').rglob('*.trace.json.gz'))
    if not traces:
        result.update(error=True,error_reason='Profile requested but no native trace was produced')
    else:
        trace_evidence=[]
        for trace in traces:
            with gzip.open(trace,'rt') as stream:
                events=json.load(stream)['traceEvents']
            kernel_count=sum(event.get('cat')=='kernel' for event in events)
            if not kernel_count:
                raise RuntimeError(f'Native trace has no GPU kernels: {trace}')
            trace_evidence.append(dict(path=str(trace),event_count=len(events),kernel_count=kernel_count))
            del events
        result['native_profile_traces']=trace_evidence
(out/'result.json').write_text(json.dumps(result,indent=2))
raise SystemExit(int(bool(result.get('error'))))
