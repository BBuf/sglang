"""Task-only BCG observation, loaded lazily in native workers; never for timing."""
import contextvars
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

directory = os.environ.get('QWEN21_BCG_AUDIT_DIR')
if directory:
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_hash != os.environ.get('QWEN21_BCG_AUDIT_SOURCE_SHA256'):
        raise RuntimeError('BCG audit source hash differs from the frozen visit')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / f'pid-{os.getpid()}.jsonl'
    active_call = contextvars.ContextVar('qwen21_bcg_audit_call', default=None)
    serial = 0
    runner_module = 'sglang.multimodal_gen.runtime.breakable_cuda_graph.runner'
    graph_module = 'sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph'

    def context():
        module = sys.modules.get('sglang.multimodal_gen.runtime.managers.forward_context')
        if module is None:
            return {}
        try:
            value = module.get_forward_context()
        except Exception:
            return {}
        batch = value.forward_batch
        return dict(request_id=getattr(batch, 'request_id', None),
                    is_warmup=getattr(batch, 'is_warmup', None),
                    quality=getattr(getattr(batch, 'sampling_params', None), 'quality', None),
                    current_timestep=value.current_timestep if isinstance(value.current_timestep, int) else None)

    def record(event, **values):
        with log_path.open('a') as stream:
            stream.write(json.dumps(dict(event=event, pid=os.getpid(), wall_ns=time.time_ns(),
                call_id=active_call.get(), context=context(), **values), default=str)+'\n')

    def signature(runner, kwargs):
        key = runner._signature(kwargs)
        return key, hashlib.sha256(repr(key).encode()).hexdigest()

    def install_runner(module):
        cls = module.BaseBreakableCudaGraphRunner
        original_capture = cls.capture
        original_call = cls.__call__
        original_replay = cls.replay

        @functools.wraps(original_capture)
        def capture(self, **kwargs):
            key, digest = signature(self, kwargs)
            existed = key in self.entries
            record('capture_enter', runner=id(self), signature=digest, already_present=existed,
                   signatures=len(self.entries))
            result = original_capture(self, **kwargs)
            entry = self.entries.get(key)
            record('capture_return', runner=id(self), signature=digest, result=result,
                   added=not existed and entry is not None, signatures=len(self.entries),
                   graph=None if entry is None else id(entry.graph),
                   segments=0 if entry is None else entry.num_segments,
                   disabled_reason=self._disabled_reason, blocked=key in self._blocked)
            return result

        @functools.wraps(original_call)
        def call(self, **kwargs):
            global serial
            serial += 1
            token = active_call.set(serial)
            key, digest = signature(self, kwargs)
            record('runner_enter', runner=id(self), signature=digest,
                   signature_summary=module._signature_summary(key),
                   entry_present=key in self.entries, signatures=len(self.entries),
                   disabled_reason=self._disabled_reason, blocked=key in self._blocked)
            try:
                output = original_call(self, **kwargs)
                record('runner_return', runner=id(self), signature=digest,
                       signatures=len(self.entries))
                return output
            except BaseException as exc:
                record('runner_error', error=repr(exc))
                raise
            finally:
                active_call.reset(token)

        @functools.wraps(original_replay)
        def replay(self, entry, kwargs):
            live = len(module._flatten_kwargs(kwargs))
            record('runner_replay_enter', runner=id(self), graph=id(entry.graph),
                   live_leaves=live, static_leaves=len(entry.static_leaves),
                   leaf_mismatch_fallback=live != len(entry.static_leaves))
            return original_replay(self, entry, kwargs)

        cls.capture, cls.__call__, cls.replay = capture, call, replay

    def install_graph(module):
        cls = module.BreakableCUDAGraph
        original = cls.replay

        @functools.wraps(original)
        def replay(self):
            record('graph_replay_enter', graph=id(self), segments=len(self._segments),
                   eager_breaks=len(self._break_fns))
            result = original(self)
            # The reviewed native loop calls each segment.replay once before
            # reaching this return; exceptions cannot produce this receipt.
            record('graph_replay_return', graph=id(self), segments=len(self._segments),
                   eager_breaks=len(self._break_fns))
            return result

        cls.replay = replay

    installers = {runner_module: install_runner, graph_module: install_graph}

    class Loader(importlib.abc.Loader):
        def __init__(self, original, name):
            self.original, self.name = original, name

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            installers[self.name](module)
            record('module_instrumented', module=self.name, source_hash=source_hash)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in installers:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                spec.loader = Loader(spec.loader, fullname)
            return spec

    sys.meta_path.insert(0, Finder())
    record('audit_loaded', source_hash=source_hash, benchmark=False)
