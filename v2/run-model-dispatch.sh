#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8
for item in baseline:A1 candidate:B1 candidate:B2 baseline:A2; do
    arm=${item%:*}
    label=${item#*:}
    cd /scratch/cleanup-v2/$arm
    PYTHONPATH=$PWD/python OUTPUT=../results/model-$label.json python3 ../bench_model_dispatch.py > ../results/model-$label.log 2>&1
done
echo 0 > ../results/model-benchmarks.exit
