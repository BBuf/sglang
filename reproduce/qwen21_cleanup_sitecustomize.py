"""Request-level selection of two archived native functions; diagnostic use only."""
import os

if os.environ.get('QWEN_CLEANUP_MARKER'):
    import ast
    import functools
    import hashlib
    import inspect
    import json
    import time
    from pathlib import Path

    from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as model
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage

    cls = model.QwenImage21FeedForward
    native = cls.forward
    root = Path('/task/evidence/preflight')
    manifest = json.loads((root/'qwen21-cleanup-source-equivalence.json').read_text())
    functions = {}
    for version, label in [('v3', 'precleanup'), ('v4', 'final')]:
        path = root/f'qwen21-cleanup-source-{version}.py'
        source = path.read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == manifest['sources'][version]
        definition = next(n for n in ast.parse(source).body
                          if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
        function = next(n for n in definition.body
                        if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        assert hashlib.sha256(ast.unparse(function).encode()).hexdigest() == manifest['functions'][version]
        namespace = dict(native.__globals__)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        functions[label] = namespace['forward']
    installed = Path(inspect.getsourcefile(cls)).read_bytes()
    assert hashlib.sha256(installed).hexdigest() == manifest['sources']['v4']
    original_stage = DenoisingStage.forward
    hook_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    @functools.wraps(original_stage)
    def observed_stage(self, batch, server_args):
        if type(self).__name__ != 'QwenImage21DenoisingStage':
            return original_stage(self, batch, server_args)
        marker = Path(os.environ['QWEN_CLEANUP_MARKER'])
        variant = marker.read_text().strip() if marker.exists() else 'final'
        assert variant in functions
        assert not server_args.enable_breakable_cuda_graph
        previous = cls.forward
        cls.forward = functions[variant]
        record = dict(request_id=batch.request_id, variant=variant, is_warmup=bool(batch.is_warmup),
                      width=int(batch.width), height=int(batch.height), pid=os.getpid(),
                      quality=str(batch.sampling_params.quality), hook_sha256=hook_sha,
                      function_sha256=manifest['functions']['v3' if variant=='precleanup' else 'v4'],
                      started_ns=time.time_ns())
        try:
            return original_stage(self, batch, server_args)
        finally:
            cls.forward = previous
            record['finished_ns'] = time.time_ns()
            folder = Path(os.environ['QWEN_CLEANUP_AUDIT_DIR'])
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/f'worker-{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')

    DenoisingStage.forward = observed_stage
