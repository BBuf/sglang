#!/bin/bash
cd /scratch/cleanup-v2/candidate
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8 PYTHONPATH=$PWD/python
python3 -m pytest -q test/registered/kernels/ops/diffusion/test_complex_rope.py test/registered/kernels/ops/diffusion/test_rmsnorm_preserve_reduction.py > ../results/final-layout-tests.log 2>&1
status=$?
echo $status > ../results/final-layout-tests.exit
if [ "$status" -eq 0 ]; then
    bash /scratch/cleanup-v2/run-benchmarks.sh
fi
