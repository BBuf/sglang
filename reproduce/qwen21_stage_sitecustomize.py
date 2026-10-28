"""Opt-in metadata observer for separate native Omni pipeline-profiler runs only."""
import os

if os.environ.get('QWEN_STAGE_AUDIT_DIR'):
    import functools
    import hashlib
    import json
    import time
    from pathlib import Path

    import torch
    import vllm_omni.diffusion.profiler.diffusion_pipeline_profiler as profiling

    original_profiler = profiling.profiler

    def tensor_metadata(value):
        if isinstance(value, torch.Tensor):
            return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
        return dict(type=type(value).__name__)

    def precision_profiler(name, func, instance):
        timed = original_profiler(name, func, instance)
        if name != 'QwenImage21Pipeline.diffuse':
            return timed

        @functools.wraps(timed)
        def observed_diffuse(*args, **kwargs):
            record = dict(pid=os.getpid(), name=name,
                          started_ns=time.time_ns(),
                          hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          inputs={key:tensor_metadata(value) for key,value in kwargs.items()},
                          positional_inputs=[tensor_metadata(value) for value in args],
                          cuda_autocast=torch.is_autocast_enabled('cuda'), scheduler_calls=[])
            scheduler = instance.scheduler
            original_step = scheduler.step
            had_override = 'step' in scheduler.__dict__
            previous_override = scheduler.__dict__.get('step')

            @functools.wraps(original_step)
            def observed_step(*step_args, **step_kwargs):
                noise = step_kwargs.get('model_output', step_args[0] if step_args else None)
                sample = step_kwargs.get('sample', step_args[2] if len(step_args)>2 else None)
                output = original_step(*step_args, **step_kwargs)
                result = output[0] if isinstance(output, tuple) else output.prev_sample
                record['scheduler_calls'].append(dict(model_output=tensor_metadata(noise),
                    sample=tensor_metadata(sample), result=tensor_metadata(result),
                    cuda_autocast=torch.is_autocast_enabled('cuda')))
                return output

            scheduler.step = observed_step
            try:
                output = timed(*args, **kwargs)
                record['output'] = tensor_metadata(output)
                return output
            finally:
                if had_override:
                    scheduler.step = previous_override
                else:
                    del scheduler.step
                # No tensor reads, copies or synchronizations were added. Write
                # metadata after the native diffuse stage timer has stopped.
                folder = Path(os.environ['QWEN_STAGE_AUDIT_DIR'])
                record['finished_ns'] = time.time_ns()
                folder.mkdir(parents=True, exist_ok=True)
                with (folder/f'omni-worker-{os.getpid()}.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')

        return observed_diffuse

    profiling.profiler = precision_profiler
