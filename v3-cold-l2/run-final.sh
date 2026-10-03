#!/bin/bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8
root=/scratch/cold-l2
# Wait only on the preceding job owned by this task; never overlap GPU runs.
while [ ! -f "$root/results/process.exit" ]; do sleep 10; done
mkdir -p "$root/final-results"
python3 "$root/verify_sources_final.py" || exit 1
for arm in baseline candidate candidate-final; do
  cd "$root/$arm"
  PYTHONPATH="$PWD/python" python3 -m pytest -q python/sglang/multimodal_gen/test/unit/test_wan_temb_table_slices.py > "$root/final-results/$arm-wan.log" 2>&1
  echo $? > "$root/final-results/$arm-wan.exit"
done
cd "$root/candidate-final"
PYTHONPATH="$PWD/python" python3 -m pytest -q test/registered/kernels/ops/diffusion test/registered/unit/kernels/test_kernels_namespace.py > "$root/final-results/candidate-tests.log" 2>&1
echo $? > "$root/final-results/candidate-tests.exit"
for item in baseline:A1 candidate-final:B1 candidate-final:B2 baseline:A2; do
  arm=${item%:*}; label=${item#*:}
  cd "$root/$arm"
  PYTHONPATH="$PWD/python" ARM=$arm OUTPUT="$root/final-results/cold-$label.json" python3 "$root/bench_cold_l2_final.py" > "$root/final-results/cold-$label.log" 2>&1 || { echo $? > "$root/final-results/benchmark-failed.exit"; exit 1; }
done
mkdir -p "$root/final-results/eager"
for item in baseline:A1 candidate-final:B1 candidate-final:B2 baseline:A2; do
  arm=${item%:*}; label=${item#*:}
  cd "$root/$arm"
  PYTHONPATH="$PWD/python" ARM=$arm OUTPUT="$root/final-results/eager/cold-$label.json" python3 "$root/bench_cold_eager.py" > "$root/final-results/eager/cold-$label.log" 2>&1 || { echo $? > "$root/final-results/eager/benchmark-failed.exit"; exit 1; }
done
cd "$root/candidate-final"
PYTHONPATH="$PWD/python" python3 -m pytest -q python/sglang/multimodal_gen/test/unit > "$root/final-results/model-tests.log" 2>&1
echo $? > "$root/final-results/model-tests.exit"
echo 0 > "$root/final-results/all-done.exit"
