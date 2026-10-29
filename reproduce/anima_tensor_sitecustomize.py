"""Opt-in observation of frozen native Anima requests; never a timing run."""
import os

if os.environ.get("ANIMA_TENSOR_AUDIT_DIR"):
    import functools
    import hashlib
    import importlib.abc
    import importlib.machinery
    import json
    import sys
    import time
    from pathlib import Path

    root = Path(os.environ["ANIMA_TENSOR_AUDIT_DIR"])
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

    def sgl_rope(module):
        original = module._anima_rope
        @functools.wraps(original)
        def rope(q, k, cos, sin):
            capture = counts.get('sgl-rope-input', 0) == 0
            values = cpu(dict(q=q, k=k, cos=cos, sin=sin)) if capture else None
            output = original(q, k, cos, sin)
            if capture:
                save('sgl-rope-input', **values, output=output, gate_disabled=module._ANIMA_ROPE.disabled)
            return output
        module._anima_rope = rope

    def scheduler(module):
        cls = module.FlowMatchEulerDiscreteScheduler
        original = cls.step
        @functools.wraps(original)
        def step(self, *args, **kwargs):
            index = self.step_index
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
        'sglang.multimodal_gen.runtime.models.encoders.qwen3': lambda m: wrap_forward(m.Qwen3ForCausalLM, 'sgl-text-encoder'),
        'transformers.models.qwen3.modeling_qwen3': lambda m: wrap_forward(m.Qwen3Model, 'omni-text-encoder'),
        'sglang.multimodal_gen.runtime.models.adapter.anima': lambda m: wrap_forward(m.AnimaTextConditioner, 'sgl-conditioner'),
        'vllm_omni.diffusion.models.anima.anima_text_conditioner': lambda m: wrap_forward(m.AnimaTextConditioner, 'omni-conditioner'),
        'sglang.multimodal_gen.runtime.models.dits.anima': sgl_rope,
        'sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete': scheduler,
        'diffusers.schedulers.scheduling_flow_match_euler_discrete': scheduler,
    }

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
