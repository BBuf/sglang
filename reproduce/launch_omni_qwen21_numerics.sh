#!/bin/sh
set -eu
cd /task/diagnostics/qwen21-numerics-omni
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(git rev-parse HEAD)" = 817d4c17feb702289a2d2b2cbacfd594c262b7a7
export PYTHONPATH=/task/diagnostics/qwen21-numerics-omni:/task
export QWEN21_DIAGNOSTIC_DIR=/task/outputs/diagnostics/qwen21-numerics1-omni
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_HOME=/task/cache/qwen21-hf
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TRITON_CACHE_DIR=/task/cache/triton-omni
export TORCHINDUCTOR_CACHE_DIR=/task/cache/inductor-omni
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
date -u
exec vllm serve /task/cache/common/Qwen-Image-2.1 --omni --num-gpus 1 \
  --dtype bfloat16 --host 127.0.0.1 --port 31051
