#!/usr/bin/env bash
set -euo pipefail
exec 9>/campaign/gpu.lock
flock 9
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=/campaign/candidate-lingbot-current/python TRITON_DISABLE_LINE_INFO=0
run=/campaign/profile/lingbot-fp32-norm-h200
python - <<'GPU_CHECK'
import subprocess,time
selected=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
for _ in range(30):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if not any(line.startswith(selected) for line in rows.splitlines()):break
    time.sleep(1)
else:raise RuntimeError('Assigned GPU remains busy before NCU; no process killed')
GPU_CHECK
for site in norm1 cross_norm; do
  for provider in baseline fused; do
    ncu --profile-from-start off --set full --section PmSampling --section PmSampling_WarpStates -o "$run/reports/full-$site-$provider" python "$run/harness/run_kernel.py" --provider "$provider" --site "$site"
    ncu --profile-from-start off --set source --section SourceCounters -o "$run/reports/source-$site-$provider" python "$run/harness/run_kernel.py" --provider "$provider" --site "$site"
    ncu --import "$run/reports/full-$site-$provider.ncu-rep" --page details > "$run/analysis/details-$site-$provider.txt"
  done
done
