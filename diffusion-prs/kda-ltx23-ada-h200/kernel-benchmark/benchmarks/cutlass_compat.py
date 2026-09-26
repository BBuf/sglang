"""Narrow compatibility helpers for the locked CUDA 13 CuTe-DSL stack.

The frozen GDN prefill baseline uses the CUDA 12.9 enum spelling of
``cute.arch.fence_proxy``.  The CUDA 13 build of cutlass-dsl 4.5.2 exposes the
same operation through string literals instead.  Install the two missing enum
names and translate their values without changing the frozen baseline source.
"""

from __future__ import annotations

import inspect

_OVERLAY_ERROR = """The CUDA 13 CUTLASS-DSL overlay is incomplete.
The generated Python wrappers and native MLIR extension do not have matching
APIs. Rebuild this workspace with ./setup_workspace.sh before running kernels.
"""


def _require_coherent_cu13_overlay() -> None:
    """Fail early instead of surfacing unrelated MLIR errors during compilation."""

    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    opview_doc = ir.OpView.__init__.__doc__ or ""
    dtors_parameters = inspect.signature(llvm.mlir_global_dtors).parameters
    if "name: str" not in opview_doc or "data" not in dtors_parameters:
        raise RuntimeError(_OVERLAY_ERROR)


def install_cutlass_452_cu13_compat() -> None:
    """Make the frozen enum-style GDN baseline work with the CUDA 13 API."""

    from cutlass import cute

    if getattr(cute.arch, "_gdn_fence_proxy_compat", False):
        return

    _require_coherent_cu13_overlay()

    # Do not wrap a coherent future wheel that restores the enum exports.
    if hasattr(cute.arch, "ProxyKind") and hasattr(cute.arch, "SharedSpace"):
        cute.arch._gdn_fence_proxy_compat = True
        return

    from cutlass._mlir.dialects.nvvm import ProxyKind, SharedSpace

    original = cute.arch.fence_proxy
    supports_use_intrinsic = "use_intrinsic" in inspect.signature(original).parameters

    def fence_proxy(kind, *, space=None, use_intrinsic=None, loc=None, ip=None):
        if isinstance(kind, ProxyKind):
            kind = str(kind)
        if isinstance(space, SharedSpace):
            space = str(space)

        kwargs = {"space": space, "loc": loc, "ip": ip}
        if supports_use_intrinsic:
            kwargs["use_intrinsic"] = use_intrinsic
        return original(kind, **kwargs)

    cute.arch.ProxyKind = ProxyKind
    cute.arch.SharedSpace = SharedSpace
    cute.arch.fence_proxy = fence_proxy
    cute.arch._gdn_fence_proxy_compat = True
