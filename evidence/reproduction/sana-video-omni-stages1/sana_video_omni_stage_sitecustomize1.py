"""Enable native synchronized stage timers in an isolated diagnostic process."""
import os

if os.environ.get('SANA_VIDEO_STAGE_AUDIT') == '1':
    import functools
    import importlib.abc
    import importlib.machinery
    import sys

    target = 'vllm_omni.diffusion.profiler.diffusion_pipeline_profiler'

    def install(module):
        original = module.DiffusionPipelineProfilerMixin.setup_diffusion_pipeline_profiler

        @functools.wraps(original)
        def setup(self, profiler_targets=None, enable_diffusion_pipeline_profiler=False):
            if self.__class__.__name__ == 'SanaVideoPipeline' and enable_diffusion_pipeline_profiler:
                profiler_targets = ['forward', 'encode_prompt', 'diffuse', 'vae.decode']
                print('[SanaVideoStageAudit] native synchronized timers: '+repr(profiler_targets), flush=True)
            return original(self, profiler_targets, enable_diffusion_pipeline_profiler)

        module.DiffusionPipelineProfilerMixin.setup_diffusion_pipeline_profiler = setup

    class Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner
        def create_module(self, spec):
            return self.inner.create_module(spec)
        def exec_module(self, module):
            self.inner.exec_module(module)
            install(module)
        def __getattr__(self, name):
            return getattr(self.inner, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != 'vllm_omni.diffusion.profiler.diffusion_pipeline_profiler':
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())
