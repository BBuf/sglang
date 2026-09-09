"""Saved-output replay correctness; run each eager/BCG and TP1/TP2 cell separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', required=True, choices=['3', '3,6'])
    parser.add_argument('--bcg', action='store_true')
    args = parser.parse_args()
    root = Path('/data/goals/sglang-h200-diffusion-20260907')
    repo = root/'flux-bcg-final-repo'
    revision = '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'
    cache = root/'model-caches/flux-ion9-torch213-initial-lossless-eager/huggingface'
    checkpoint = cache/'hub/models--black-forest-labs--FLUX.1-dev/snapshots'/revision
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == '3d51468a54b8fad047b1d679c0968ac2919613e7'
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    selected = set(args.gpus.split(','))
    deadline = time.monotonic() + 30
    while True:
        gpu_state = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        idle = True
        for line in gpu_state.splitlines():
            index, uuid, memory, util = [x.strip() for x in line.split(',')]
            if index in selected:
                assert int(memory) < 256, line
                idle = idle and int(util) == 0
        if idle:
            break
        # NVML's utilization sampling window may still cover the process that
        # just exited even after its memory has returned to zero.
        assert time.monotonic() < deadline, gpu_state
        time.sleep(1)
    token = Path('/tmp/h200-diffusion-hf-token').read_text().strip()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, HF_HOME=str(cache), HF_HUB_OFFLINE='1',
                      HF_TOKEN=token, HUGGINGFACE_HUB_TOKEN=token, FLASHINFER_DISABLE_VERSION_CHECK='1',
                      SGLANG_DIFFUSION_SYNC_STAGE_PROFILING='1', NCCL_NVLS_ENABLE='0', CI='1')
    import sglang
    import torch
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    assert Path(sglang.__file__).resolve() == repo/'python/sglang/__init__.py'
    assert torch.__version__ == '2.13.0+cu130'
    label = f'flux-ion9-bcg-replay-v1-tp{len(selected)}-' + ('bcg' if args.bcg else 'eager')
    output = root/'results'/label
    output.mkdir(exist_ok=False)
    state = dict(pid=os.getpid(), started_at=time.time(), source_commit=commit, checkpoint=revision,
                 gpus=args.gpus, initial_gpu_state=gpu_state, torch=torch.__version__, bcg=args.bcg, rows=[])
    path = output/'result.json'
    generator = None
    # Revisit a previous request after switching both prompt and seed. The last
    # shape is deliberately not warmed and must exercise the existing fallback.
    cases = [
        ('city', 'A futuristic cyberpunk city at night, neon lights reflecting on wet streets', 1024, 1024, 42),
        ('cube', 'A red cube on a white table, centered product photo', 512, 512, 7),
        ('butterfly', 'A blue butterfly flying above green grass, watercolor illustration', 512, 512, 123),
        ('cube-repeat', 'A red cube on a white table, centered product photo', 512, 512, 7),
        ('landscape', 'An alpine lake surrounded by snowy mountains, morning light, oil painting', 768, 512, 9),
        ('unwarmed', 'A yellow ceramic teapot on a wooden table', 640, 512, 19),
    ]
    try:
        generator = DiffGenerator.from_pretrained(
            model_path=str(checkpoint), model_id='black-forest-labs/FLUX.1-dev', revision=revision,
            backend='sglang', num_gpus=len(selected), tp_size=len(selected),
            component_residency={'dit': 'resident'}, enable_torch_compile=False,
            enable_breakable_cuda_graph=args.bcg, warmup_mode='server',
            warmup_resolutions=['1024x1024', '512x512', '768x512'],
            warmup_sampling_params={'num_inference_steps': 2}, master_port=29675,
        )
        for name, prompt, width, height, seed in cases:
            print('REPLAY_REQUEST_START', name, flush=True)
            result = generator.generate(dict(prompt=prompt, width=width, height=height, seed=seed,
                                             num_inference_steps=6, guidance_scale=3.5, quality='lossless',
                                             save_output=True, output_path=str(output), output_file_name=f'{name}.png'))
            assert result is not None and not isinstance(result, list), name
            image = Path(result.output_file_path)
            assert image.is_file()
            state['rows'].append(dict(name=name, prompt=prompt, width=width, height=height, seed=seed,
                                      steps=6, output=str(image), sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                                      metrics=result.metrics, peak_memory_mb=result.peak_memory_mb))
            path.write_text(json.dumps(state, indent=2, default=str)+'\n')
            print('REPLAY_REQUEST_END', name, state['rows'][-1]['sha256'], flush=True)
        assert state['rows'][1]['sha256'] == state['rows'][3]['sha256'], 'stale static inputs after prompt/seed switch'
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'failed'
        state['exception'] = repr(exc)
        raise
    finally:
        if generator is not None:
            generator.shutdown()
        state['finished_at'] = time.time()
        path.write_text(json.dumps(state, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
