#!/bin/bash
set -uo pipefail
root=/scratch/cold-l2
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8
while [ ! -f "$root/final-results/attention-done.exit" ]; do sleep 10; done
mkdir -p "$root/reviewed-results/more-models" "$root/reviewed-results/stream" "$root/reviewed-results/eager" "$root/reviewed-results/nsys"
python3 "$root/verify_sources_reviewed.py" || exit 1
cd "$root/candidate-reviewed"
PYTHONPATH="$PWD/python" python3 -m pytest -q python/sglang/multimodal_gen/test/unit/test_ideogram_rope_swiglu_fusion.py python/sglang/multimodal_gen/test/unit/test_ernie_rope_geglu_fusion.py python/sglang/multimodal_gen/test/unit/test_wan_temb_table_slices.py > "$root/reviewed-results/focused-tests.log" 2>&1
echo $? > "$root/reviewed-results/focused-tests.exit"
for item in baseline:A1 candidate-reviewed:B1 candidate-reviewed:B2 baseline:A2; do
 arm=${item%:*}; label=${item#*:}; cd "$root/$arm"
 PYTHONPATH="$PWD/python" ARM=$arm OUTPUT="$root/reviewed-results/more-models/cold-$label.json" python3 "$root/bench_more_models.py" > "$root/reviewed-results/more-models/cold-$label.log" 2>&1 || exit 1
done
echo 0 > "$root/reviewed-results/models-done.exit"
cd "$root/candidate-reviewed"
PYTHONPATH="$PWD/python" python3 -m pytest -q test/registered/kernels/ops/diffusion test/registered/unit/kernels/test_kernels_namespace.py > "$root/reviewed-results/kernel-tests.log" 2>&1
echo $? > "$root/reviewed-results/kernel-tests.exit"
for mode in stream eager; do
 for item in baseline:A1 candidate-reviewed:B1 candidate-reviewed:B2 baseline:A2; do
  arm=${item%:*}; label=${item#*:}; cd "$root/$arm"
  PYTHONPATH="$PWD/python" ARM=$arm OUTPUT="$root/reviewed-results/$mode/cold-$label.json" python3 "$root/bench_cold_$mode.py" > "$root/reviewed-results/$mode/cold-$label.log" 2>&1 || exit 1
 done
done
for arm in baseline candidate-reviewed; do
 cd "$root/$arm"
 PYTHONPATH="$PWD/python" ARM=$arm FILTER=residual/flux_text/torch.float16,residual/flux2_joint/torch.float16 OUTPUT="$root/reviewed-results/nsys/$arm.json" nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node -f true -o "$root/reviewed-results/nsys/$arm" python3 "$root/bench_cold_l2_final.py" > "$root/reviewed-results/nsys/$arm.log" 2>&1 || exit 1
 nsys stats --report cuda_gpu_kern_gb_sum --format csv "$root/reviewed-results/nsys/$arm.nsys-rep" > "$root/reviewed-results/nsys/$arm-kernels.csv" 2>&1
done
echo 0 > "$root/reviewed-results/done.exit"
