#!/usr/bin/env bash
set -euo pipefail
exec 9>/campaign/gpu.lock
flock 9
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=/campaign/candidate-longlive2-wan-norm/python
run=/campaign/profile/longlive2-wan-norm-post-h200
for provider in baseline fused; do
  ncu --profile-from-start off --set full --section PmSampling --section PmSampling_WarpStates -o "$run/reports/full-$provider" python "$run/harness/run_kernel.py" --provider "$provider" --channels-last
  ncu --profile-from-start off --set source --section SourceCounters -o "$run/reports/source-$provider" python "$run/harness/run_kernel.py" --provider "$provider" --channels-last
  ncu --import "$run/reports/full-$provider.ncu-rep" --page details > "$run/analysis/details-$provider.txt"
done
