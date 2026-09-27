"""Explicit validation-only LTX audio reproduction context, symmetric in A/B."""
import functools
import json
import os
from pathlib import Path
import runpy

if os.environ.get('KDA_DETERMINISTIC_AUDIO_DECODE') == '1':
    import torch
    # Establish normal native initialization order before importing model classes.
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
    from sglang.multimodal_gen.runtime.models.vaes.ltx_2_audio import AutoencoderKLLTX2Audio
    from sglang.multimodal_gen.runtime.models.vocoder.ltx_2_vocoder import LTX2Vocoder

    def audio_context(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            b = torch.backends
            saved = (b.cudnn.allow_tf32, b.cuda.matmul.allow_tf32,
                     b.cudnn.deterministic, b.cudnn.benchmark)
            b.cudnn.allow_tf32 = False
            b.cuda.matmul.allow_tf32 = False
            b.cudnn.deterministic = True
            b.cudnn.benchmark = False
            try:
                return fn(*args, **kwargs)
            finally:
                (b.cudnn.allow_tf32, b.cuda.matmul.allow_tf32,
                 b.cudnn.deterministic, b.cudnn.benchmark) = saved
                directory = Path(os.environ['KDA_REPRO_DIR'])
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / f'audio-context-{os.getpid()}.jsonl').open('a') as out:
                    out.write(json.dumps({'method': fn.__qualname__, 'cudnn_deterministic': True,
                                          'cudnn_benchmark': False, 'cudnn_tf32': False,
                                          'matmul_tf32': False, 'restored': True}) + '\n')
        return wrapped

    AutoencoderKLLTX2Audio.decode = audio_context(AutoencoderKLLTX2Audio.decode)
    LTX2Vocoder.forward = audio_context(LTX2Vocoder.forward)

if os.environ.get('KDA_CAPTURE_DIR'):
    runpy.run_path(str(Path(__file__).resolve().parent.parent / 'capture/sitecustomize.py'))
