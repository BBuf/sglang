#!/bin/bash
cd /scratch/cleanup-v2/candidate
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8 PYTHONPATH=$PWD/python
python3 -m pytest -q test/registered/kernels/ops/diffusion test/registered/unit/kernels/test_kernels_namespace.py > ../results/candidate-expanded.log 2>&1
echo $? > ../results/candidate-expanded.exit
