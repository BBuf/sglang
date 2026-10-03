#!/bin/bash
set -uo pipefail
root=/scratch/cold-l2
while [ ! -f "$root/final-results/all-done.exit" ]; do sleep 10; done
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8
mkdir -p "$root/final-results/stream"
for item in baseline:A1 candidate-final:B1 candidate-final:B2 baseline:A2; do
 arm=${item%:*}; label=${item#*:}
 cd "$root/$arm"
 PYTHONPATH="$PWD/python" ARM=$arm OUTPUT="$root/final-results/stream/cold-$label.json" python3 "$root/bench_cold_stream.py" > "$root/final-results/stream/cold-$label.log" 2>&1 || { echo $? > "$root/final-results/stream/failed.exit"; exit 1; }
done
for arm in baseline candidate-final; do
 cd "$root/$arm"
 PYTHONPATH="$PWD/python" OUTPUT="$root/final-results/stream/graph-probe-$arm.json" python3 "$root/probe_graph_variance.py" > "$root/final-results/stream/graph-probe-$arm.log" 2>&1 || exit 1
done
echo 0 > "$root/final-results/stream/done.exit"
