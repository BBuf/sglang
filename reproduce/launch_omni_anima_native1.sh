#!/bin/sh
set -eu
cd /task/repos/vllm-omni
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(git rev-parse HEAD)" = 817d4c17feb702289a2d2b2cbacfd594c262b7a7
test -z "$(git diff HEAD --)"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_HOME=/task/cache/anima-hf
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TRITON_CACHE_DIR=/task/cache/triton-omni
export TORCHINDUCTOR_CACHE_DIR=/task/cache/inductor-omni
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
date -u
exec vllm serve /task/cache/common/Anima-Base-v1.0-official/split_files/diffusion_models/anima-base-v1.0.safetensors --omni --num-gpus 1 \
  --model-class-name AnimaPipeline \
  --custom-pipeline-args '{"components_path":"/task/cache/common/Anima-Base-v1.0-Diffusers"}' \
  --dtype bfloat16 --host 127.0.0.1 --port 31051
