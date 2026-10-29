"""Opt-in counters for separate Anima quality/replay diagnostics; no tensor reads."""
import os

if os.environ.get('ANIMA_BCG_AUDIT_DIR'):
    import collections
    import functools
    import hashlib
    import json
    import time
    from pathlib import Path

    from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import BaseBreakableCudaGraphRunner
    from sglang.multimodal_gen.runtime.models.dits import anima
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage

    active = None
    hook_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_stage = DenoisingStage.forward
    original_model = anima.AnimaTransformer3DModel.forward
    original_replay = BaseBreakableCudaGraphRunner.replay
    original_capture = BaseBreakableCudaGraphRunner.capture
    original_build = BaseBreakableCudaGraphRunner._capture

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

    @functools.wraps(original_build)
    def build(self, kwargs, key):
        start = time.perf_counter()
        entry = original_build(self, kwargs, key)
        if active is not None:
            active['actual_captures'].append(dict(wall_seconds=time.perf_counter()-start, segments=entry.num_segments))
        return entry

    @functools.wraps(original_stage)
    def stage_forward(self, batch, server_args):
        global active
        previous = active
        active = dict(pid=os.getpid(), hook_sha256=hook_sha, request_id=batch.request_id,
                      is_warmup=bool(batch.is_warmup), quality=str(batch.sampling_params.quality),
                      width=int(batch.width), height=int(batch.height),
                      started_ns=time.time_ns(), python_model_forwards=0, replay_calls=0,
                      capture_calls=0, actual_captures=[], segments_per_replay=collections.Counter(),
                      guidance_scale=float(batch.guidance_scale))
        try:
            return original_stage(self, batch, server_args)
        finally:
            record = active
            active = previous
            record['finished_ns'] = time.time_ns()
            record['rope_gate'] = dict(disabled=anima._ANIMA_ROPE.disabled, verified=anima._ANIMA_ROPE.verified, signatures=[str(x) for x in anima._ANIMA_ROPE.verified_sigs])
            record['mounted_quality'] = str(self._mounted_quality)
            record['runners'] = [dict(quality=str(key[1]), entries=len(runner.entries))
                                 for key, runner in self._bcg_runners.items()]
            folder = Path(os.environ['ANIMA_BCG_AUDIT_DIR'])
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/f'sgl-worker-{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')

    DenoisingStage.forward = stage_forward
    anima.AnimaTransformer3DModel.forward = model_forward
    BaseBreakableCudaGraphRunner.replay = replay
    BaseBreakableCudaGraphRunner.capture = capture

    BaseBreakableCudaGraphRunner._capture = build
