#!/bin/sh
set -eu
cd "$SGL_REPO"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(git rev-parse HEAD)" = "$SGL_EXPECTED_HEAD"
git diff --check
export PYTHONPATH="$SGL_REPO/python"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_HOME=/task/cache/anima-hf
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0
export TRITON_CACHE_DIR=/task/cache/triton-sgl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
date -u
exec sglang serve --model-path /task/cache/common/Anima-Base-v1.0-Diffusers \
  --model-id Anima-Base-v1.0-Diffusers --backend sglang \
  --num-gpus 1 --tp-size 1 --ulysses-degree 1 --ring-degree 1 \
  --performance-mode manual --dit-cpu-offload false --dit-layerwise-offload false \
  --text-encoder-cpu-offload false --vae-cpu-offload false \
  --enable-torch-compile false --enable-breakable-cuda-graph false \
  --warmup-resolutions 1024x1024 --warmup-num-frames 1 --warmup-steps 50 \
  --attention-backend torch_cudnn_sdpa --host 127.0.0.1 --port 31050
