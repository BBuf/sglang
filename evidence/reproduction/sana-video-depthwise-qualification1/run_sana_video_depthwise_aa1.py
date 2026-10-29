"""Run the pinned A/A plan once, then audit saved artifacts and noise."""
import subprocess
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
subprocess.run(['python3', str(root/'run_sana_video_blocks1.py'), str(root/'sana-video-depthwise-main-eager-aa1.json')], check=True)
subprocess.run(['docker', 'exec', '-e', 'CUDA_VISIBLE_DEVICES=', 'bbuf-diffusion-rtx6000-20261026-sgl', 'python3', '/task/admit_sana_video_depthwise_aa1.py'], check=True)
