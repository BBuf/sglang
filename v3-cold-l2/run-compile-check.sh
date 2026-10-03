#!/bin/bash
set -uo pipefail
root=/scratch/cold-l2
while [ ! -f "$root/reviewed-results/done.exit" ]; do sleep 10; done
cd "$root/candidate-reviewed"
CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8 PYTHONPATH="$PWD/python" OUTPUT="$root/reviewed-results/ideogram-graph-compile.json" python3 "$root/check_ideogram_graph_compile.py" > "$root/reviewed-results/ideogram-graph-compile.log" 2>&1
echo $? > "$root/reviewed-results/compile-check.exit"
