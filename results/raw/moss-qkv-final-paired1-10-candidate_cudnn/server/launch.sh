#!/bin/sh
set -eu
cd /task/candidates/moss-qkv
export PYTHONPATH=/task/candidates/moss-qkv/python
python python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py print-idle-gpus --count 1
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(git rev-parse HEAD)" = "${MOSS_CANDIDATE_SHA:-af21e2b989e482cfdce39689126562f2209d53ac}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_HOME=/task/cache/moss-hf
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
export TRITON_CACHE_DIR=/task/cache/triton-sgl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
date -u
exec sglang serve --model-path /task/cache/common/MOSS-Image-Base \
  --backend sglang --num-gpus 1 --tp-size 1 \
  --ulysses-degree 1 --ring-degree 1 --performance-mode speed \
  --disable-autocast true \
  --component-attention-backends transformer=torch_cudnn_sdpa \
  --enable-torch-compile false --enable-breakable-cuda-graph false \
  --warmup-resolutions 1024x1024 --warmup-steps 50 \
  --host 127.0.0.1 --port 31030
