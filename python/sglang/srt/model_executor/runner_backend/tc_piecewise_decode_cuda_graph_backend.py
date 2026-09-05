# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TcPiecewiseDecodeCudaGraphBackend — torch.compile piecewise decode graph.

Decode-time analogue of TcPiecewiseCudaGraphBackend (prefill). FX-splits the
decode forward at the attention / mamba-state metadata ops so inductor fuses
the surrounding elementwise/index/scan work and the per-step metadata
recompute is hoisted off the single-stream critical path. This is the
cross-model fix for the SM120 dispatch gap (see DECODE_PIECEWISE_PROPOSAL).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import torch
import tqdm

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.compilation.compile_phase import (
    enable_torch_compile_warmup,
    set_pcg_capture_stream,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
    _suppress_lru_cache_dynamo_warning,
    _toggle_fused_ops,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey
    from sglang.srt.server_args import ServerArgs


_VALID_COMPILERS = ("eager", "inductor")


class TcPiecewiseDecodeCudaGraphBackend(BaseCudaGraphBackend):
    """torch.compile-driven piecewise decode capture; a single compiled
    callable is reused for every decode bs bucket (torch.compile owns the
    per-shape cache), so this backend keeps no _graphs table."""

    def __init__(self, cuda_graph_runner: BaseCudaGraphRunner) -> None:
        model_runner = cuda_graph_runner.model_runner
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._compile_config: CompilationConfig = self.build_compilation_config(
            model_runner.server_args
        )
        self._language_model: torch.nn.Module = getattr(
            model_runner.model, "language_model", model_runner.model
        )
        _suppress_lru_cache_dynamo_warning()
        self._run_compile_pass(cuda_graph_runner)
        self._compiled_fn: Callable = model_runner.model.forward

    @staticmethod
    def build_compilation_config(server_args: ServerArgs) -> CompilationConfig:
        """Decode CompilationConfig: capture sizes come from the decode bs
        buckets (num_tokens = bs * captured_req_width handled by the runner)."""
        decode = get_exec().graph.cuda_graph_config.decode
        bs = decode.bs
        compiler = decode.tc_compiler
        assert bs is not None, "cuda_graph_config[decode].bs is not set"
        assert compiler in _VALID_COMPILERS, (
            f"By now, only {_VALID_COMPILERS} are supported for the "
            "tc_piecewise decode compiler."
        )
        config = CompilationConfig(
            list(bs), compiler, server_args.enable_torch_compile_debug_mode
        )
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            config.add_split_op("sglang.moe_forward_piecewise_cuda_graph_impl")
        return config

    @staticmethod
    def install_compile(
        language_model: Any,
        *,
        compile_config: CompilationConfig,
        graph_pool: Any,
        fullgraph: bool = True,
        dynamic_arg_dims: Optional[Any] = None,
    ) -> None:
        install_torch_compiled(
            language_model,
            fullgraph=fullgraph,
            dynamic_arg_dims=dynamic_arg_dims,
            compile_config=compile_config,
            graph_pool=graph_pool,
        )

    def _run_compile_pass(self, cuda_graph_runner: BaseCudaGraphRunner) -> None:
        """Install torch.compile, then drive one decode forward per bs bucket
        inside enable_torch_compile_warmup so FX/inductor sees every shape
        before any CUDA graph is captured."""
        language_model = self._language_model
        inner_model = getattr(language_model, "model", language_model)
        compiler = self._compile_config.compiler
        capture_bs = cuda_graph_runner.capture_bs
        with enable_tc_piecewise_cuda_graph():
            try:
                if compiler != "eager":
                    _toggle_fused_ops(inner_model, reverse=False, num_tokens=1)
                cuda_graph_runner._run_dummy_decode_forward(capture_bs[0])
                if self._pool is None:
                    self._pool = get_or_create_global_graph_memory_pool(
                        self._device_module
                    )
                set_graph_pool_id(self._pool)
                self.install_compile(
                    inner_model,
                    compile_config=self._compile_config,
                    graph_pool=self._pool,
                )
                with enable_torch_compile_warmup():
                    if is_hip():
                        cuda_graph_runner._run_dummy_decode_forward(capture_bs[-1])
                    else:
                        compile_range = (
                            tqdm.tqdm(list(reversed(capture_bs)))
                            if get_parallel().tp_rank == 0
                            else reversed(capture_bs)
                        )
                        for bs in compile_range:
                            if get_parallel().tp_rank == 0:
                                compile_range.set_description(
                                    f"Compiling decode ({bs=})"
                                )
                            cuda_graph_runner._run_dummy_decode_forward(bs)
            finally:
                _toggle_fused_ops(inner_model, reverse=True, num_tokens=1)

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream) -> Iterator[None]:
        self._capture_stream = stream
        try:
            with self.replay_session():
                with set_pcg_capture_stream(stream):
                    yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return True

    @contextmanager
    def replay_session(self) -> Iterator[None]:
        with enable_tc_piecewise_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        return self._compiled_fn(
            static_forward_batch.input_ids,
            static_forward_batch.positions,
            static_forward_batch,
            **kwargs,
        )

    def cleanup(self) -> None:
        self._compiled_fn = None
        self._compile_config = None
        self._language_model = None
        self._pool = None
