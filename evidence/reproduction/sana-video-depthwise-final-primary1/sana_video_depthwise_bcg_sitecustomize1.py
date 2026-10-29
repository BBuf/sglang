"""Opt-in counters for separate SANA-Video quality/replay diagnostics; no tensor reads."""
import os

if os.environ.get('SANA_VIDEO_BCG_AUDIT_DIR'):
    import collections
    import functools
    import hashlib
    import json
    import time
    from pathlib import Path

    from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import BaseBreakableCudaGraphRunner
    from sglang.multimodal_gen.runtime.models.dits import sana, sana_video
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage

    active = None
    hook_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_stage = DenoisingStage.forward
    original_replay = BaseBreakableCudaGraphRunner.replay
    original_capture = BaseBreakableCudaGraphRunner.capture
    original_build = BaseBreakableCudaGraphRunner._capture

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
                      num_frames=int(batch.num_frames), started_ns=time.time_ns(), replay_calls=0,
                      capture_calls=0, actual_captures=[], segments_per_replay=collections.Counter(),
                      guidance_scale=float(batch.guidance_scale))
        try:
            return original_stage(self, batch, server_args)
        finally:
            record = active
            active = previous
            record['finished_ns'] = time.time_ns()
            record['ln_gate'] = dict(disabled=sana._SANA_LN_MOD.disabled, wrapper_disabled=sana._SANA_LN_MOD_DISABLED, signatures=[str(x) for x in sana._SANA_LN_MOD.verified_sigs])
            gate = getattr(sana_video, '_SANA_VIDEO_DEPTHWISE_GLU', None)
            record['depthwise_gate'] = None if gate is None else dict(disabled=gate.disabled, verified=gate.verified, signatures=[str(x) for x in gate.verified_sigs])
            record['mounted_quality'] = str(self._mounted_quality)
            record['runners'] = [dict(quality=str(key[1]), entries=len(runner.entries))
                                 for key, runner in self._bcg_runners.items()]
            folder = Path(os.environ['SANA_VIDEO_BCG_AUDIT_DIR'])
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/f'sgl-worker-{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')

    DenoisingStage.forward = stage_forward
    BaseBreakableCudaGraphRunner.replay = replay
    BaseBreakableCudaGraphRunner.capture = capture

    BaseBreakableCudaGraphRunner._capture = build
