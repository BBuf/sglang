"""Opt-in observation of frozen native SANA-Video requests; never a timing run."""
import os

if os.environ.get("SANA_VIDEO_TENSOR_AUDIT_DIR"):
    import functools
    import hashlib
    import importlib.abc
    import importlib.machinery
    import json
    import sys
    import time
    from pathlib import Path

    root = Path(os.environ["SANA_VIDEO_TENSOR_AUDIT_DIR"])
    counts = {}

    def cpu(value):
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {str(k): cpu(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [cpu(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": type(value).__name__}

    def save(kind, **values):
        import torch
        index = counts.get(kind, 0)
        counts[kind] = index + 1
        folder = root / str(os.getpid())
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{kind}-{index:03d}.pt"
        assert not path.exists(), path
        torch.save(dict(kind=kind, index=index, time_ns=time.time_ns(),
                        torch=str(torch.__version__), diagnostic_only=True,
                        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        **cpu(values)), path)

    def wrap_forward(cls, kind):
        import inspect
        original = cls.forward
        signature = inspect.signature(original)
        @functools.wraps(original)
        def forward(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            values = cpu({k: v for k, v in bound.arguments.items() if k != 'self'})
            output = original(self, *args, **kwargs)
            result = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
            save(kind, inputs=values, output=result)
            return output
        cls.forward = forward

    def scheduler(module):
        cls = module.DPMSolverMultistepScheduler
        original = cls.step
        @functools.wraps(original)
        def step(self, *args, **kwargs):
            index = getattr(self, "_inner", self).step_index
            capture = index is None or index == 0 or index == 49
            values = {}
            if capture:
                values = cpu(dict(model_output=kwargs.get("model_output", args[0] if args else None),
                                  timestep=kwargs.get("timestep", args[1] if len(args)>1 else None),
                                  sample=kwargs.get("sample", args[2] if len(args)>2 else None),
                                  timesteps=self.timesteps, sigmas=self.sigmas))
            output = original(self, *args, **kwargs)
            if capture:
                save("scheduler-step", source=cls.__module__, step_index_before=index,
                     result=output[0] if isinstance(output, tuple) else output.prev_sample, **values)
            return output
        cls.step = step

    callbacks = {
        'sglang.multimodal_gen.runtime.models.encoders.gemma2': lambda m: wrap_forward(m.Gemma2Model, 'sgl-text-encoder'),
        'transformers.models.gemma2.modeling_gemma2': lambda m: wrap_forward(m.Gemma2Model, 'omni-text-encoder'),
    }
    if os.environ['SANA_VIDEO_TENSOR_AUDIT_ARM'] == 'sgl':
        callbacks['sglang.multimodal_gen.runtime.models.schedulers.scheduling_dpm_solver_multistep'] = scheduler
    else:
        callbacks['diffusers.schedulers.scheduling_dpmsolver_multistep'] = scheduler


    class Loader(importlib.abc.Loader):
        def __init__(self, inner, callback):
            self.inner, self.callback = inner, callback
        def create_module(self, spec):
            return self.inner.create_module(spec)
        def exec_module(self, module):
            self.inner.exec_module(module)
            self.callback(module)
        def __getattr__(self, name):
            return getattr(self.inner, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in callbacks:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader, callbacks[fullname])
            return spec

    sys.meta_path.insert(0, Finder())
