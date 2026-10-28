"""Opt-in counters for separate Qwen quality/replay diagnostics; no tensor reads."""
import os

if os.environ.get('QWEN_BCG_AUDIT_DIR'):
    import collections
    import functools
    import hashlib
    import json
    import time
    from pathlib import Path

    from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import BaseBreakableCudaGraphRunner
    from sglang.multimodal_gen.runtime.models.dits import qwen_image21
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage

    active = None
    hook_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_stage = DenoisingStage.forward
    original_model = qwen_image21.QwenImage21Transformer2DModel.forward
    original_replay = BaseBreakableCudaGraphRunner.replay
    original_capture = BaseBreakableCudaGraphRunner.capture

    @functools.wraps(original_model)
    def model_forward(self, *args, **kwargs):
        if active is not None:
            active['python_model_forwards'] += 1
        return original_model(self, *args, **kwargs)

    @functools.wraps(original_replay)
    def replay(self, entry, kwargs):
        if active is not None:
            active['replay_calls'] += 1
            active['segments_per_replay'][str(entry.num_segments)] += 1
        return original_replay(self, entry, kwargs)

    @functools.wraps(original_capture)
    def capture(self, **kwargs):
        if active is not None:
            active['capture_calls'] += 1
        return original_capture(self, **kwargs)

    original_packed = getattr(qwen_image21, 'fused_packed_silu_mul_bitexact', None)
    if original_packed is not None:
        @functools.wraps(original_packed)
        def packed(x, *args, **kwargs):
            if active is not None:
                active['packed_python_calls'][str(tuple(x.shape))] += 1
            return original_packed(x, *args, **kwargs)
        qwen_image21.fused_packed_silu_mul_bitexact = packed

    @functools.wraps(original_stage)
    def stage_forward(self, batch, server_args):
        global active
        if type(self).__name__ != 'QwenImage21DenoisingStage':
            return original_stage(self, batch, server_args)
        previous = active
        active = dict(pid=os.getpid(), hook_sha256=hook_sha, request_id=batch.request_id,
                      is_warmup=bool(batch.is_warmup), quality=str(batch.sampling_params.quality),
                      width=int(batch.width), height=int(batch.height),
                      started_ns=time.time_ns(), python_model_forwards=0, replay_calls=0,
                      capture_calls=0, segments_per_replay=collections.Counter(),
                      packed_python_calls=collections.Counter())
        try:
            return original_stage(self, batch, server_args)
        finally:
            record = active
            active = previous
            record['finished_ns'] = time.time_ns()
            record['mounted_quality'] = str(self._mounted_quality)
            record['runners'] = [dict(quality=str(key[1]), entries=len(runner.entries))
                                 for key, runner in self._bcg_runners.items()]
            folder = Path(os.environ['QWEN_BCG_AUDIT_DIR'])
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/f'sgl-worker-{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')

    DenoisingStage.forward = stage_forward
    qwen_image21.QwenImage21Transformer2DModel.forward = model_forward
    BaseBreakableCudaGraphRunner.replay = replay
    BaseBreakableCudaGraphRunner.capture = capture
