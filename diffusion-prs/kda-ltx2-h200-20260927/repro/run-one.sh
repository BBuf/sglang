#!/usr/bin/env bash
set -euo pipefail
: "${SGLANG_REPO:?Set to baseline or candidate checkout}"
: "${LTX_MODEL_PATH:?Set to the complete LTX-2.3 checkpoint}"
: "${LTX_OUTPUT_DIR:?Set to a new output directory}"
: "${CUDA_VISIBLE_DEVICES:?Choose one idle H200}"
ltx_repro_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$LTX_OUTPUT_DIR"
export PYTHONPATH="$ltx_repro_dir:$SGLANG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export KDA_DETERMINISTIC_AUDIO_DECODE=1
export KDA_REPRO_DIR="$LTX_OUTPUT_DIR/audio-context"
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
sglang \
  generate \
  --backend=sglang \
  "--model-path=${LTX_MODEL_PATH}" \
  --revision=7caa482d5cd10a2eae6b34cb48f093ebc45a263e \
  '--prompt=A beautiful sunset over the ocean' \
  --seed=42 \
  --width=768 \
  --height=512 \
  --num-frames=121 \
  --num-inference-steps=30 \
  --guidance-scale=3 \
  --num-gpus=1 \
  --tp-size=1 \
  --ulysses-degree=1 \
  --master-port=32030 \
  --scheduler-port=32130 \
  --port=32230 \
  --performance-mode=manual \
  --enable-torch-compile=false \
  --dit-cpu-offload=false \
  --dit-layerwise-offload=false \
  --vae-cpu-offload=false \
  --text-encoder-cpu-offload=false \
  --quality=high \
  --warmup-mode=request \
  --warmup-resolutions=768x512 \
  --warmup-num-frames=121 \
  --save-output \
  "--output-path=${LTX_OUTPUT_DIR}" \
  --output-file-name=sample \
  "--perf-dump-path=${LTX_OUTPUT_DIR}/perf.json" \
  --pipeline-class-name=LTX2Pipeline \
  --fps=24
