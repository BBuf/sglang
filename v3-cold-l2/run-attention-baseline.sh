#!/bin/bash
set -uo pipefail
root=/scratch/cold-l2
while [ ! -f "$root/final-results/diagnostics.exit" ]; do sleep 10; done
for arm in baseline candidate-final; do
 cd "$root/$arm"
 CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8 PYTHONPATH="$PWD/python" python3 -m pytest -q python/sglang/multimodal_gen/test/unit/test_subblock_sparse_attention.py > "$root/final-results/$arm-attention.log" 2>&1
 echo $? > "$root/final-results/$arm-attention.exit"
done
echo 0 > "$root/final-results/attention-done.exit"
