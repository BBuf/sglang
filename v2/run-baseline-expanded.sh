#!/bin/bash
cd /scratch/cleanup-v2/baseline
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8 PYTHONPATH=$PWD/python
python3 -m pytest -q test/registered/kernels/ops/diffusion test/registered/unit/kernels/test_kernels_namespace.py > ../results/baseline-expanded.log 2>&1
echo $? > ../results/baseline-expanded.exit
