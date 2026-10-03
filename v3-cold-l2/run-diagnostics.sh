#!/bin/bash
set -uo pipefail
root=/scratch/cold-l2
while [ ! -f "$root/final-results/stream/done.exit" ]; do sleep 10; done
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8
mkdir -p "$root/final-results/more-models" "$root/final-results/nsys"
for item in baseline:A1 candidate-final:B1 candidate-final:B2 baseline:A2; do
 arm=${item%:*}; label=${item#*:}
 cd "$root/$arm"
 PYTHONPATH="$PWD/python" ARM=$arm OUTPUT="$root/final-results/more-models/cold-$label.json" python3 "$root/bench_more_models.py" > "$root/final-results/more-models/cold-$label.log" 2>&1 || { echo $? > "$root/final-results/more-models/failed.exit"; exit 1; }
done
for arm in baseline candidate-final; do
 cd "$root/$arm"
 PYTHONPATH="$PWD/python" ARM=$arm FILTER=qknorm_complex_triton/17,silu_mul/17,complex_rope/17,cat_pad/8/6/6 OUTPUT="$root/final-results/nsys/$arm.json" nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node -f true -o "$root/final-results/nsys/$arm" python3 "$root/bench_cold_l2_final.py" > "$root/final-results/nsys/$arm.log" 2>&1 || exit 1
 nsys stats --report cuda_gpu_kern_sum --format csv "$root/final-results/nsys/$arm.nsys-rep" > "$root/final-results/nsys/$arm-kernels.csv" 2>&1
done
echo 0 > "$root/final-results/diagnostics.exit"
