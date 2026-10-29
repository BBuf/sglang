#!/bin/sh
set -eu
cd /task/candidates/sana-video-depthwise-glu-v1
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(git rev-parse HEAD)" = af21e2b989e482cfdce39689126562f2209d53ac
test "$(git diff HEAD --binary --full-index | sha256sum | cut -d' ' -f1)" = 6a073c8df5e246ea84485ac999a3eefc0b2c8949f67b682a85c579d46b345291
export PYTHONPATH=/task/candidates/sana-video-depthwise-glu-v1/python
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_HOME=/task/cache/sana-video-hf
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0
export TRITON_CACHE_DIR=/task/cache/triton-sgl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
date -u
exec sglang serve --model-path /task/cache/common/SANA-Video_2B_480p_diffusers \
  --model-id SANA-Video_2B_480p_diffusers --backend sglang \
  --num-gpus 1 --tp-size 1 --ulysses-degree 1 --ring-degree 1 \
  --performance-mode manual --dit-cpu-offload false --dit-layerwise-offload false \
  --text-encoder-cpu-offload false --vae-cpu-offload false \
  --enable-torch-compile false --enable-breakable-cuda-graph false \
  --warmup-resolutions 832x480 --warmup-num-frames 81 --warmup-steps 50 \
  --host 127.0.0.1 --port 31050
