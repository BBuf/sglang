# KDA provenance: scaffolded for the Humanize2 / Kernel Design Agents SM120
# NVFP4 MoE sessions (kda_sessions/sm120_nvfp4_moe/DESIGN.md).
#
# SM120 NVFP4 block-scaled MoE grouped-GEMM for Qwen3.6-35B-A3B-NVFP4.
#
# Structure:
#   - MoeWorkList: host-side (expert, m_tile, n_tile) tile scheduler state,
#     built from the moe_align_block_size expert offsets. This is the grouped
#     dimension the dense kernel does not have.
#   - _Qwen3xNvfp4MoeSm120Kernel: grouped fork of the proven dense kernel
#     (_Qwen3xNvfp4Sm120Kernel in qwen3x_nvfp4_gemm_sm120.py). It reuses the
#     MmaMXF4NVF4Op m16n8k64 mainloop, TMA producer warp, and blockscaled
#     SFA/SFB smem layouts verbatim, and overrides the tile scheduler plus
#     the B/SFB/C pointer derivation for per-expert weights.
#
# v1 status: SCAFFOLD. Compile-clean, registration wired, but the grouped
# launch raises NotImplementedError so callers fall back to
# flashinfer_cutlass. The optimization sessions fill in the TODO(OPT-n)
# hooks in order; see DESIGN.md section 7.
#
# OPT-2 status: the grouped fork (_Qwen3xNvfp4MoeSm120Kernel) is the real
# single-launch path. Each CTA of a plain grid is one (expert, m_tile,
# n_tile) work item from MoeWorkList; B/SFB are 4D (E, N, K/2) TMA tensors
# with the L mode = expert, selected by coordinate per work item (one
# descriptor pair total). SwiGLU / requantize / router-weighted scatter stay
# in torch between the two grouped launches (OPT-3 fuses them).

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Int32, cute
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo
from cutlass import pipeline, utils
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.cute.runtime import make_ptr

from sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm_sm120 import (
    _Qwen3xNvfp4Sm120Kernel,
    _max_active_clusters,
)

_SF_VEC_SIZE = 16
_TILE_M = 16  # dense kernel CTA tile M (one warp-MMA atom row)
_TILE_N = 64  # dense kernel CTA tile N
# Max work items for one grouped launch: the moe-align sort pads each
# expert's rows to the _TILE_M tile, so num_tiles <= num_experts * ceil(n /
# _TILE_N) + (extra rows from block padding) * ceil(n / _TILE_N). 256
# experts * 16 n-tiles (gemm1, N=1024) = 4096; round up for headroom.
_MAX_TILES = 8192
# Production expert count for nvidia/Qwen3.6-35B-A3B-NVFP4; keyed as a
# constexpr at compile so the grouped launch keeps a single B descriptor.
_NUM_EXPERTS = 256
# Padded-M upper bound for one grouped launch: M_total = bs * top_k.
_MAX_M = 48 * 8  # decode-graph bs ceiling 48 * top_k 8

# Per-expert (K, N) for Qwen3.6-35B-A3B-NVFP4 (hidden=2048, moe_inter=512).
# gemm1 is the fused gate_up projection consumed as two 512-wide halves.
_GEMM1_KN = (2048, 1024)  # hidden -> gate(512) || up(512)
_GEMM2_KN = (512, 2048)  # moe_inter -> hidden
_SUPPORTED_GEMM_SHAPES = frozenset({_GEMM1_KN, _GEMM2_KN, (2048, 512)})


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class MoeWorkList:
    """Grouped tile scheduler state: flat (expert, m_tile, n_tile) tiles.

    Tokens are pre-sorted by expert (moe_align_block_size), so expert e owns
    the contiguous row range [expert_m_offsets[e], expert_m_offsets[e+1]).
    Each CTA tile is one _TILE_M-row slice of one expert's rows crossed with
    one _TILE_N-column slice; tiles never straddle experts, so the dense
    mainloop runs unmodified per tile with per-expert B/SFB pointers.

    Attributes:
        tile_expert: int32 [num_tiles], expert id per tile (indexes B/SFB).
        tile_m_row: int32 [num_tiles], first sorted row of the tile.
        tile_m_rows: int32 [num_tiles], live rows in the tile (<= _TILE_M).
        tile_n_col: int32 [num_tiles], first output column of the tile.
        num_tiles: total tiles; grid.x for the persistent grouped launch.
    """

    tile_expert: torch.Tensor
    tile_m_row: torch.Tensor
    tile_m_rows: torch.Tensor
    tile_n_col: torch.Tensor
    num_tiles: int


def build_moe_work_list(
    expert_m_offsets: torch.Tensor,
    n: int,
    *,
    tile_m: int = _TILE_M,
    tile_n: int = _TILE_N,
) -> MoeWorkList:
    """Build the flat (expert, m_tile, n_tile) work list.

    Args:
        expert_m_offsets: int32/int64 [num_experts + 1] row offsets into the
            sorted activation (prefix sums of per-expert token counts), as
            produced by the align/sort step upstream of the MoE runner.
        n: GEMM N for this projection (1024 gate_up, 2048 down).

    TODO(OPT-2): build this on-device inside the align kernel (or a 1-CTA
    prologue) so the grouped launch stays fully async and CUDA-graph safe;
    the host loop below forces a sync through expert_m_offsets.cpu().
    """
    offsets = expert_m_offsets.to(torch.int64).cpu()
    counts = offsets[1:] - offsets[:-1]
    n_tiles_n = _ceil_div(n, tile_n)
    experts, m_rows, m_counts, n_cols = [], [], [], []
    for expert_id in range(counts.shape[0]):
        rows = int(counts[expert_id])
        if rows == 0:
            continue
        base = int(offsets[expert_id])
        for m_tile in range(_ceil_div(rows, tile_m)):
            live = min(tile_m, rows - m_tile * tile_m)
            for n_tile in range(n_tiles_n):
                experts.append(expert_id)
                m_rows.append(base + m_tile * tile_m)
                m_counts.append(live)
                n_cols.append(n_tile * tile_n)
    device = expert_m_offsets.device
    as_tensor = lambda vals: torch.tensor(vals, dtype=torch.int32, device=device)
    return MoeWorkList(
        tile_expert=as_tensor(experts),
        tile_m_row=as_tensor(m_rows),
        tile_m_rows=as_tensor(m_counts),
        tile_n_col=as_tensor(n_cols),
        num_tiles=len(experts),
    )


class _Qwen3xNvfp4MoeSm120Kernel(_Qwen3xNvfp4Sm120Kernel):
    """OPT-2 grouped NVFP4 MoE kernel: dense SM120 mainloop + expert dim.

    One launch over the MoeWorkList; each CTA is one (expert, m_tile,
    n_tile) work item and tiles never straddle experts. The dense mainloop
    (MmaMXF4NVF4Op m16n8k64, TMA producer warp, blockscaled SFA/SFB smem,
    manual mma_atom unroll) is inherited unchanged; the grouped fork
    overrides only the tile scheduler / coordinate mapping:

    - ``_compute_grid``: plain grid, grid.x = MoeWorkList.num_tiles
      (ctaid == work-item index). The work list is built n-tile-major
      within an expert for SFB/L2 reuse; experts are ordered by the
      moe-align sort.
    - ``__call__`` builds B/SFB as 4D (E, N, K) / (32, 4, sf_n, 4, sf_k, E)
      ordered layouts so one TMA descriptor pair covers every expert; the
      expert is selected by the L-mode coordinate per work item.
    - ``_run_grouped_producer`` / ``_run_grouped_consumer`` are the dense
      per-tile bodies with the (m, n, l) tile coordinate mapped through the
      work-list tensors loaded in the prologue. Every ``@cute.jit`` call
      site has its own constant-folded copy keyed on the constexpr flags,
      so per-CTA branches inside the kernel body (``is_valid_tile``,
      per-expert alpha) stay dynamic.
    """

    def __init__(self, device_index: int):
        super().__init__(
            direct_scheduler=False, m1_epilogue=False, cache_policy=False
        )
        self.device_index = device_index
        self._compiled: dict[tuple, object] = {}
        self._compile_lock = threading.Lock()

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
        direct_one_m_tile_scheduler: bool,
        num_tiles: int,
    ) -> tuple:
        # Plain grouped grid: ctaid.x == work-item index. The problem shape
        # recorded here is only used for the direct_scheduler validity check,
        # which the grouped path does not take.
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, (1, 1, 1)
        )
        for source_name, runtime_name in (
            ("raster_along_m", "_raster_along_m"),
            ("cluster_shape_major_fdd", "cluster_shape_m_fdd"),
            ("cluster_shape_minor_fdd", "cluster_shape_n_fdd"),
        ):
            if not hasattr(tile_sched_params, source_name) and hasattr(
                tile_sched_params, runtime_name
            ):
                object.__setattr__(
                    tile_sched_params,
                    source_name,
                    getattr(tile_sched_params, runtime_name),
                )
        return tile_sched_params, (num_tiles, 1, 1)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        tile_expert: cute.Tensor,
        tile_m_row: cute.Tensor,
        tile_n_col: cute.Tensor,
        alphas: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the grouped GEMM over the prebuilt work list.

        a/b/c/sfa/sfb are the fully-formed cute tensors built by `wrapper`
        (dense-style: a/b/c are (M,K,1)/(N,K,1)/(M,N,1) fp4/bf16, sfa/sfb are
        6D swizzled). tile_expert/tile_m_row/tile_n_col are the int32
        MoeWorkList tensors; num_tiles is read from tile_expert.shape[0].
        alphas is fp32 [E]. Mirrors the dense kernel's __call__ setup.
        """
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        a_tensor = a
        b_tensor = b
        c_tensor = c

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a_tensor,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b_tensor,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            self.sfa_tile_shape_mk,
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c_tensor,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        num_tiles = cute.size(tile_expert, mode=[0])
        tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.tile_shape_mnk,
            max_active_clusters,
            self.direct_one_m_tile_scheduler,
            num_tiles,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            a_tensor,
            tma_atom_b,
            tma_tensor_b,
            b_tensor,
            tma_atom_sfa,
            tma_tensor_sfa,
            sfa_tensor,
            tma_atom_sfb,
            tma_tensor_sfb,
            sfb_tensor,
            tma_atom_c,
            tma_tensor_c,
            c_tensor,
            tile_expert,
            tile_m_row,
            tile_n_col,
            alphas,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        tile_expert: cute.Tensor,
        tile_m_row: cute.Tensor,
        tile_n_col: cute.Tensor,
        alphas: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
    ):
        """Dense-style compile entry: build the 3D/6D cute tensors, then call.

        mA is (sym_m, K/2) uint8, mB is the expert-folded (E*N, K/2) uint8,
        mC is (sym_m, N) bf16. sf_m/sf_n/sf_k follow the dense convention
        (sf_m over A rows, sf_n over the *folded* B rows so the SFB atom
        covers every expert's swizzled scales, sf_k over K). The 6D SFA/SFB
        ordered layouts are identical to the dense wrapper.
        """
        m = cute.size(mA, mode=[0])
        k_raw = cute.size(mA, mode=[1])
        bn = cute.size(mB, mode=[0])
        n = cute.size(mC, mode=[1])
        k = k_raw * 2

        a_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
        b_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((bn, k, 1), order=(1, 0, 2)),
        )
        c_tensor = cute.make_tensor(
            mC.iterator,
            layout=cute.make_ordered_layout((m, n, 1), order=(1, 0, 2)),
        )
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_m, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )

        self(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            tile_expert,
            tile_m_row,
            tile_n_col,
            alphas,
            max_active_clusters,
            current_stream,
        )

    def compile(self, kn: tuple[int, int], max_m: int):
        """Compile the grouped kernel for one (K, N) projection.

        Fake tensors: A uint8 (sym_m, K/2) k-contiguous with
        sf_m = ceil(max_m / 128) swizzled SFA rows, B uint8
        (num_experts, N, K/2) k-contiguous, SFB swizzled
        (E, N_pad, K_pad), C bf16 (sym_m, N) row-major, plus the
        MoeWorkList int32 tensors and the fp32 [E] alphas.
        """
        key = (kn, max_m)
        compiled = self._compiled.get(key)
        if compiled is not None:
            return compiled

        with self._compile_lock, torch.cuda.device(self.device_index):
            compiled = self._compiled.get(key)
            if compiled is not None:
                return compiled

            k, n = kn
            num_experts = _NUM_EXPERTS
            sf_m = _ceil_div(max_m, 128)
            # SFB covers every expert's swizzled scales, so sf_n spans the
            # expert-folded B rows (E * n).
            sf_n = _ceil_div(num_experts * n, 128)
            sf_k = _ceil_div(k // _SF_VEC_SIZE, 4)

            sym_m = cute.sym_int()
            a_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Uint8, (sym_m, k // 2),
                stride_order=(1, 0), assumed_align=32,
            )
            sym_bn = cute.sym_int()
            b_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Uint8, (sym_bn, k // 2),
                stride_order=(1, 0), assumed_align=32,
            )
            # SFA/SFB are passed as raw gmem pointers (dense-kernel style);
            # the kernel rebuilds the swizzled layout from the A/B shapes.
            sfa_fake = make_ptr(
                cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16
            )
            sfb_fake = make_ptr(
                cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16
            )
            c_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.BFloat16, (sym_m, n),
                stride_order=(1, 0), assumed_align=16,
            )
            sym_tiles = cute.sym_int()
            w_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_tiles,),
                stride_order=(0,), assumed_align=16,
            )
            alpha_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32, (num_experts,), assumed_align=16,
            )
            stream_fake = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True
            )
            max_active_clusters = _max_active_clusters(self.device_index)

            compiled = cute.compile(
                self.wrapper,
                a_fake,
                b_fake,
                c_fake,
                sf_m,
                sf_n,
                sf_k,
                sfa_fake,
                sfb_fake,
                w_fake,
                w_fake,
                w_fake,
                alpha_fake,
                max_active_clusters,
                stream_fake,
                options="--opt-level 2 --enable-tvm-ffi",
            )
            self._compiled[key] = compiled
            return compiled

    # -- grouped per-tile bodies -------------------------------------------
    # These are the dense per-tile bodies with the (m, n, l) tile coordinate
    # mapped through the work list; each call site is constant-folded for the
    # flag combination it is invoked under.

    @cute.jit
    def _run_grouped_producer(
        self,
        work_tile,
        tma_atom_a,
        tAgA,
        tAsA,
        tma_atom_sfa,
        tAgSFA,
        tAsSFA,
        tma_atom_b,
        tBgB,
        tBsB,
        tma_atom_sfb,
        tBgSFB,
        tBsSFB,
        sSFA,
        sA,
        mSFA_mkl,
        mA_mkl,
        mSFB_nkl,
        mB_nkl,
        mainloop_pipeline,
        mainloop_producer_state,
        k_tile_iter_cnt,
        k_tile_start,
        tile_expert: cute.Tensor,
        tile_m_row: cute.Tensor,
        tile_n_col: cute.Tensor,
        tile_idx: Int32,
        gemm_n: Int32,
    ):
        moe_tile_m = tile_m_row[tile_idx] // Int32(self.tile_shape_mnk[0])
        # B/SFB fold the expert dim into N (L=1): the global n-tile index is
        # expert * (gemm_n / tile_n) + local n-tile. The L coordinate is 0.
        moe_tile_n = (
            tile_expert[tile_idx] * gemm_n + tile_n_col[tile_idx]
        ) // Int32(self.tile_shape_mnk[1])

        if cutlass.const_expr(
            self.load_path == "tma" and not self.use_m1_non_tma_a
        ):
            tAgA_mkl = tAgA[(None, moe_tile_m, None, 0)]
        if cutlass.const_expr(self.load_path == "tma"):
            tBgB_nkl = tBgB[(None, moe_tile_n, None, 0)]
        if cutlass.const_expr(
            self.load_path == "tma" and not self.use_m1_non_tma_sfa
        ):
            sfa_tile_coord_m = moe_tile_m // Int32(self.sfa_tiles_per_block)
            tAgSFA_mkl = tAgSFA[(None, sfa_tile_coord_m, None, 0)]
        if cutlass.const_expr(self.load_path == "tma"):
            sfb_tile_coord_n = moe_tile_n // Int32(self.sfb_tiles_per_block)
            tBgSFB_nkl = tBgSFB[(None, sfb_tile_coord_n, None, 0)]
        if cutlass.const_expr(self.load_path == "cpasync"):
            cpasync_sfa_tile_coord_m = (
                moe_tile_m // Int32(self.sfa_tiles_per_block)
            )
            cpasync_sfb_tile_coord_n = (
                moe_tile_n // Int32(self.sfb_tiles_per_block)
            )

        mainloop_producer_state.reset_count()

        for _k_tile in range(
            0, k_tile_iter_cnt, 1, unroll=self.k_loop_unroll
        ):  # type: ignore[call-overload]
            mainloop_pipeline.producer_acquire(mainloop_producer_state)

            k_tile_global = k_tile_start + mainloop_producer_state.count
            if cutlass.const_expr(self.load_path == "tma"):
                tBgB_k = tBgB_nkl[(None, k_tile_global)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                if cutlass.const_expr(not self.use_m1_non_tma_a):
                    tAgA_k = tAgA_mkl[(None, k_tile_global)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tAgSFA_k = tAgSFA_mkl[(None, k_tile_global)]
                    tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                tBgSFB_k = tBgSFB_nkl[(None, k_tile_global)]
                tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

            if cutlass.const_expr(self.load_path == "cpasync"):
                pass
            elif cutlass.const_expr(self.use_m1_non_tma_a):
                lane = Int32(cute.arch.thread_idx()[0] % 32)
                for a_iter in cutlass.range_constexpr(
                    (self.tile_shape_mnk[2] + 31) // 32
                ):
                    k_local = lane + Int32(a_iter * 32)
                    if k_local < Int32(self.tile_shape_mnk[2]):
                        k_coord = (
                            k_tile_global * Int32(self.tile_shape_mnk[2])
                            + k_local
                        )
                        sA[
                            (Int32(0), k_local, mainloop_producer_state.index)
                        ] = mA_mkl[(Int32(0), k_coord, 0)]
            else:
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )

            if cutlass.const_expr(self.load_path == "cpasync"):
                pass
            elif cutlass.const_expr(self.use_m1_non_tma_sfa):
                lane = Int32(cute.arch.thread_idx()[0] % 32)
                scale_groups_per_k_tile = (
                    self.tile_shape_mnk[2] // self.sf_vec_size
                )
                sfa_slots = self.sfa_tile_shape_mk[0] * scale_groups_per_k_tile
                for sfa_iter in cutlass.range_constexpr(
                    (sfa_slots + 31) // 32
                ):
                    linear = lane + Int32(sfa_iter * 32)
                    m_local = linear // Int32(scale_groups_per_k_tile)
                    scale_group = linear - m_local * Int32(
                        scale_groups_per_k_tile
                    )
                    k_local_sfa = scale_group * Int32(self.sf_vec_size)
                    k_coord_sfa = (
                        k_tile_global * Int32(self.tile_shape_mnk[2])
                        + k_local_sfa
                    )
                    if linear < Int32(sfa_slots):
                        sSFA[
                            (
                                m_local,
                                k_local_sfa,
                                mainloop_producer_state.index,
                            )
                        ] = mSFA_mkl[(Int32(0), k_coord_sfa, 0)]
            else:
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA_k,
                    tAsSFA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )

            if cutlass.const_expr(self.load_path == "tma"):
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB_k,
                    tBsSFB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
            if cutlass.const_expr(self.load_path == "cpasync"):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            mainloop_pipeline.producer_commit(mainloop_producer_state)
            mainloop_producer_state.advance()

    @cute.jit
    def _run_grouped_consumer(
        self,
        work_tile,
        gC_mnl,
        sC,
        sSFA,
        sSFB,
        accumulators,
        tiled_mma,
        mma_atom,
        tma_atom_c,
        mainloop_pipeline,
        mainloop_consumer_state,
        k_tile_iter_cnt,
        epilogue_op,
        alpha_value,
        tCsA_copy_view,
        tCsB_copy_view,
        tCsSFA_copy_view_full,
        tCsSFB_copy_view_full,
        tCrA_copy_view,
        tCrB_copy_view,
        tCrSFA_copy_view_full,
        tCrSFB_copy_view_full,
        tCrA,
        tCrB,
        tCrSFA_full,
        tCrSFB_full,
        smem_tiled_copy_A,
        smem_tiled_copy_B,
        smem_tiled_copy_SFA,
        smem_tiled_copy_SFB,
        tidx,
        warp_idx,
    ):
        tile_coord_mnl = work_tile.tile_idx
        gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
        sfa_tile_offset = tile_coord_mnl[0] % self.sfa_tiles_per_block
        sfb_tile_offset = tile_coord_mnl[1] % self.sfb_tiles_per_block
        if cutlass.const_expr(self.sfa_tiles_per_block > 1):
            sSFA_tile = cute.local_tile(
                sSFA,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (sfa_tile_offset, 0, None),
            )
            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(
                sSFA_tile
            )
            tCrSFA_tile = self._partition_fragment_SFA(
                sSFA_tile[None, None, 0], tiled_mma.get_slice(tidx), tidx
            )
            tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(
                tCrSFA_tile
            )
        else:
            tCsSFA_tile_copy_view = tCsSFA_copy_view_full
            tCrSFA_tile_copy_view = tCrSFA_copy_view_full
        if cutlass.const_expr(self.sfb_tiles_per_block > 1):
            sSFB_tile = cute.local_tile(
                sSFB,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (sfb_tile_offset, 0, None),
            )
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(
                sSFB_tile
            )
            tCrSFB_tile = self._partition_fragment_SFB(
                sSFB_tile[None, None, 0], tiled_mma.get_slice(tidx), tidx
            )
            tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(
                tCrSFB_tile
            )
        else:
            tCsSFB_tile_copy_view = tCsSFB_copy_view_full
            tCrSFB_tile_copy_view = tCrSFB_copy_view_full
        accumulators.fill(0.0)

        # Pipelined MAINLOOP (identical to the dense kernel).
        mainloop_consumer_state.reset_count()

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_state.count < k_tile_iter_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_state
            )

        mainloop_pipeline.consumer_wait(
            mainloop_consumer_state, peek_ab_full_status
        )
        tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
        tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
        tCsSFA_p = tCsSFA_tile_copy_view[
            None, None, None, mainloop_consumer_state.index
        ]
        tCsSFB_p = tCsSFB_tile_copy_view[
            None, None, None, mainloop_consumer_state.index
        ]
        cute.copy(
            smem_tiled_copy_A,
            tCsA_p[None, None, 0],
            tCrA_copy_view[None, None, 0],
        )
        cute.copy(
            smem_tiled_copy_B,
            tCsB_p[None, None, 0],
            tCrB_copy_view[None, None, 0],
        )

        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

        cute.copy(
            smem_tiled_copy_SFA,
            tCsSFA_p_filtered,
            tCrSFA_copy_view_filtered,
        )
        cute.copy(
            smem_tiled_copy_SFB,
            tCsSFB_p_filtered,
            tCrSFB_copy_view_filtered,
        )

        num_k_blocks = cute.size(tCrA, mode=[2])
        for _k_tile in range(
            0, k_tile_iter_cnt - 1, 1, unroll=self.k_loop_unroll
        ):  # type: ignore[call-overload]
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_next = (
                    0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                )

                if k_block_idx == num_k_blocks - 1:
                    mainloop_pipeline.consumer_release(mainloop_consumer_state)
                    mainloop_consumer_state.advance()

                    peek_ab_full_status = cutlass.Boolean(1)
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                    tCsA_p = tCsA_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                    tCsB_p = tCsB_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                    tCsSFA_p = tCsSFA_tile_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                    tCsSFB_p = tCsSFB_tile_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]

                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, k_block_next],
                    tCrA_copy_view[None, None, k_block_next],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, k_block_next],
                    tCrB_copy_view[None, None, k_block_next],
                )

                if k_block_idx == 0:
                    tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                    tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                    tCrSFA_copy_view_filtered = cute.filter_zeros(
                        tCrSFA_tile_copy_view
                    )
                    tCrSFB_copy_view_filtered = cute.filter_zeros(
                        tCrSFB_tile_copy_view
                    )
                    cute.copy(
                        smem_tiled_copy_SFA,
                        tCsSFA_p_filtered,
                        tCrSFA_copy_view_filtered,
                    )
                    cute.copy(
                        smem_tiled_copy_SFB,
                        tCsSFB_p_filtered,
                        tCrSFB_copy_view_filtered,
                    )

                for _mt in range(self.num_m_tiles):
                    for _nt in range(self.num_n_tiles):
                        # Set the scale factors for the MMA instruction
                        mma_atom.set(
                            WarpField.SFA,
                            tCrSFA_tile[None, _mt, k_block_idx].iterator,
                        )
                        mma_atom.set(
                            WarpField.SFB,
                            tCrSFB_tile[None, _nt, k_block_idx].iterator,
                        )
                        # Unrolled MMA execution (avoids cute.gemm overhead).
                        cute.gemm(
                            mma_atom,
                            accumulators[None, _mt, _nt],
                            tCrA[None, _mt, k_block_idx],
                            tCrB[None, _nt, k_block_idx],
                            accumulators[None, _mt, _nt],
                        )

        # Remainder k tile (last iteration) with pipeline wait.
        if k_tile_iter_cnt > 0:
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_state, peek_ab_full_status
            )
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_next = (
                    0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                )

                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, k_block_next],
                    tCrA_copy_view[None, None, k_block_next],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, k_block_next],
                    tCrB_copy_view[None, None, k_block_next],
                )

                if k_block_idx == 0:
                    tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                    tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                    tCrSFA_copy_view_filtered = cute.filter_zeros(
                        tCrSFA_tile_copy_view
                    )
                    tCrSFB_copy_view_filtered = cute.filter_zeros(
                        tCrSFB_tile_copy_view
                    )
                    cute.copy(
                        smem_tiled_copy_SFA,
                        tCsSFA_p_filtered,
                        tCrSFA_copy_view_filtered,
                    )
                    cute.copy(
                        smem_tiled_copy_SFB,
                        tCsSFB_p_filtered,
                        tCrSFB_copy_view_filtered,
                    )

                for _mt in range(self.num_m_tiles):
                    for _nt in range(self.num_n_tiles):
                        mma_atom.set(
                            WarpField.SFA,
                            tCrSFA_tile[None, _mt, k_block_idx].iterator,
                        )
                        mma_atom.set(
                            WarpField.SFB,
                            tCrSFB_tile[None, _nt, k_block_idx].iterator,
                        )
                        cute.gemm(
                            mma_atom,
                            accumulators[None, _mt, _nt],
                            tCrA[None, _mt, k_block_idx],
                            tCrB[None, _nt, k_block_idx],
                            accumulators[None, _mt, _nt],
                        )

            mainloop_pipeline.consumer_release(mainloop_consumer_state)
            mainloop_consumer_state.advance()

        # EPILOGUE (identical to the dense non-swap path).
        _is_m_major = self.c_layout.is_m_major_c()
        if cutlass.const_expr(self.c_dtype.width == 16):
            copy_atom_r2s = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                self.c_dtype,
            )
        else:
            copy_atom_r2s = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
            )

        if cutlass.const_expr(self.c_dtype.width == 16):
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(), 2
                ),
                self.c_dtype,
            )
        else:
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.c_dtype
            )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
            copy_atom_C, tiled_mma
        )

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tcgc_for_tma_partition = cute.zipped_divide(
            gC_mnl_slice, self.epi_tile
        )

        bSG_sD, bSG_gD = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tcgc_for_tma_partition,
        )

        epi_rest_m = bSG_gD.shape[1][0]
        epi_rest_n = bSG_gD.shape[1][1]
        epi_tile_m = self.epi_tile[0]
        epi_tile_n = self.epi_tile[1]
        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
        has_multi_epi_store = cutlass.const_expr(
            not (self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1)
        )
        tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_mma_warps * self.num_threads_per_warp,
        )
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=tma_store_producer_group,
        )

        for epi_m in cutlass.range_constexpr(epi_rest_m):
            for epi_n in cutlass.range_constexpr(epi_rest_n):
                MmaMPerEpiM = epi_tile_m // mma_tile_m
                MmaNPerEpiN = epi_tile_n // mma_tile_n
                for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                    for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                        mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                        mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                        tRS_rD_slice = tRS_rD[
                            (None, mma_m_in_epi, mma_n_in_epi)
                        ]
                        tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                        for elem_idx in cutlass.range_constexpr(
                            cute.size(tRS_rD_slice)
                        ):
                            tRS_rD_slice[elem_idx] = tRS_rAcc_slice[
                                elem_idx
                            ]

                gmem_coord = (epi_m, epi_n)
                tRS_rD_out = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, self.c_dtype
                )
                acc_vec = tRS_rD.load()
                acc_vec = epilogue_op(
                    (alpha_value * acc_vec).to(self.c_dtype)
                )
                tRS_rD_out.store(acc_vec)

                epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                    tRS_sD, mode=[3]
                )
                if has_multi_epi_store:
                    self.epilog_sync_barrier.arrive_and_wait()
                cute.copy(
                    tiled_copy_r2s,
                    tRS_rD_out,
                    tRS_sD[(None, None, None, epi_buffer)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()

                if warp_idx == 0:
                    cute.copy(
                        tma_atom_c,
                        bSG_sD[(None, epi_buffer)],
                        bSG_gD[(None, gmem_coord)],
                    )
                    if has_multi_epi_store:
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

        if has_multi_epi_store:
            tma_store_pipeline.producer_tail()

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        directA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        directB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        directSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        directSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        directC_mnl: cute.Tensor,
        tile_expert: cute.Tensor,
        tile_m_row: cute.Tensor,
        tile_n_col: cute.Tensor,
        alphas: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            if cutlass.const_expr(
                self.load_path == "tma" and not self.use_m1_non_tma_a
            ):
                cpasync.prefetch_descriptor(tma_atom_a)
            if cutlass.const_expr(self.load_path == "tma"):
                cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(
                self.load_path == "tma" and not self.use_m1_non_tma_sfa
            ):
                cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(self.load_path == "tma"):
                cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(not self.use_m1_non_tma_c):
                cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        if cutlass.const_expr(self.use_m1_non_tma_sfa):
            tma_copy_bytes = cute.size_in_bytes(
                self.b_dtype, b_smem_layout
            ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            if cutlass.const_expr(not self.use_m1_non_tma_a):
                tma_copy_bytes += cute.size_in_bytes(
                    self.a_dtype, a_smem_layout
                )
        else:
            tma_copy_bytes = (
                cute.size_in_bytes(self.a_dtype, a_smem_layout)
                + cute.size_in_bytes(self.b_dtype, b_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            )

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline setup
        mainloop_pipeline_array_ptr = (
            storage.mainloop_pipeline_array_ptr.data_ptr()
        )
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        if cutlass.const_expr(self.load_path == "cpasync"):
            mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_threads_per_warp,
            )
            mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps * self.num_threads_per_warp,
            )
            mainloop_pipeline = pipeline.PipelineAsync.create(
                num_stages=self.ab_stage,
                producer_group=mainloop_pipeline_producer_group,
                consumer_group=mainloop_pipeline_consumer_group,
                barrier_storage=mainloop_pipeline_array_ptr,
            )
        else:
            mainloop_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.ab_stage,
                producer_group=mainloop_pipeline_producer_group,
                consumer_group=mainloop_pipeline_consumer_group,
                tx_count=tma_copy_bytes,
                barrier_storage=mainloop_pipeline_array_ptr,
                cta_layout_vmnk=cta_layout_vmnk,
            )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        if cutlass.const_expr(not self.use_m1_non_tma_sfa):
            gSFA_mkl = cute.local_tile(
                mSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            self.sfb_tile_shape_nk,
            (None, None, None),
        )
        if cutlass.const_expr(self.load_path == "cpasync"):
            gA_cpasync_mkl = cute.local_tile(
                directA_mkl,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (None, None, None),
            )
            gB_cpasync_nkl = cute.local_tile(
                directB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
            gSFA_cpasync_mkl = cute.local_tile(
                directSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
            gSFB_cpasync_nkl = cute.local_tile(
                directSFB_nkl,
                self.sfb_tile_shape_nk,
                (None, None, None),
            )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        a_cta_crd = cluster_coord_mnk[1]
        if cutlass.const_expr(
            self.load_path == "tma" and not self.use_m1_non_tma_a
        ):
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA_mkl, 0, 2),
            )

        # TMA partitions for B
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )
        b_cta_crd = cluster_coord_mnk[0]
        if cutlass.const_expr(self.load_path == "tma"):
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )

        # TMA partitions for SFA
        if cutlass.const_expr(
            self.load_path == "tma" and not self.use_m1_non_tma_sfa
        ):
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_atom_sfa,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sSFA, 0, 2),
                cute.group_modes(gSFA_mkl, 0, 2),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA partitions for SFB
        if cutlass.const_expr(self.load_path == "tma"):
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sSFB, 0, 2),
                cute.group_modes(gSFB_nkl, 0, 2),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)

        if cutlass.const_expr(self.load_path == "cpasync"):
            cpasync_tiled_copy_A = self._make_cpasync_tiled_copy(
                self.a_dtype,
                self.tile_shape_mnk[2],
            )
            cpasync_tiled_copy_B = self._make_cpasync_tiled_copy(
                self.b_dtype,
                self.tile_shape_mnk[2],
            )
            cpasync_tiled_copy_SF = self._make_scale_tiled_copy(self.sf_dtype)
            cA_mkl = cute.make_identity_tensor(cute.shape(directA_mkl))
            cA_cpasync_mkl = cute.local_tile(
                cA_mkl,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (None, None, None),
            )
            cB_nkl = cute.make_identity_tensor(cute.shape(directB_nkl))
            cB_cpasync_nkl = cute.local_tile(
                cB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
            cSFA_mkl = cute.make_identity_tensor(cute.shape(directSFA_mkl))
            cSFA_cpasync_mkl = cute.local_tile(
                cSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
            cSFB_nkl = cute.make_identity_tensor(cute.shape(directSFB_nkl))
            cSFB_cpasync_nkl = cute.local_tile(
                cSFB_nkl,
                self.sfb_tile_shape_nk,
                (None, None, None),
            )

            cpasync_lane = tidx % self.num_threads_per_warp
            thr_cpasync_A = cpasync_tiled_copy_A.get_slice(cpasync_lane)
            thr_cpasync_B = cpasync_tiled_copy_B.get_slice(cpasync_lane)
            thr_cpasync_SF = cpasync_tiled_copy_SF.get_slice(cpasync_lane)
            tAgA_cpasync_mkl = thr_cpasync_A.partition_S(gA_cpasync_mkl)
            tAsA_cpasync = thr_cpasync_A.partition_D(sA)
            tAcA_cpasync_mkl = thr_cpasync_A.partition_S(cA_cpasync_mkl)
            tBgB_cpasync_mkl = thr_cpasync_B.partition_S(gB_cpasync_mkl)
            tBsB_cpasync = thr_cpasync_B.partition_D(sB)
            tBcB_cpasync_mkl = thr_cpasync_B.partition_S(cB_cpasync_mkl)
            tAgSFA_cpasync_mkl = thr_cpasync_SF.partition_S(gSFA_cpasync_mkl)
            tAsSFA_cpasync = thr_cpasync_SF.partition_D(sSFA)
            tAcSFA_cpasync_mkl = thr_cpasync_SF.partition_S(
                cSFA_cpasync_mkl
            )
            tBgSFB_cpasync_mkl = thr_cpasync_SF.partition_S(gSFB_cpasync_mkl)
            tBsSFB_cpasync = thr_cpasync_SF.partition_D(sSFB)
            tBcSFB_cpasync_mkl = thr_cpasync_SF.partition_S(
                cSFB_cpasync_mkl
            )

        # Make fragments (grouped path never uses swap_ab).
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA_full = self._partition_fragment_SFA(
            sSFA[None, None, 0], thr_mma, tidx
        )
        tCrSFB_full = self._partition_fragment_SFB(
            sSFB[None, None, 0], thr_mma, tidx
        )
        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster/thread sync
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        block_idx = cute.arch.block_idx()
        k_tile_start = Int32(0)
        k_tile_iter_cnt = k_tile_cnt

        # Work-list lookup: one CTA == one (expert, m_tile, n_tile) item.
        tile_idx = Int32(block_idx[0])
        moe_expert = tile_expert[tile_idx]
        moe_m_row = tile_m_row[tile_idx]
        moe_n_col = tile_n_col[tile_idx]
        alpha_value = alphas[moe_expert].to(cutlass.Float32)
        gemm_n = Int32(cute.size(mC_mnl, mode=[1]))
        # Use the dense persistent scheduler path end-to-end: the grouped
        # work-list coordinate is used for B/SFB (expert fold), but the
        # mainloop work_tile / producer-consumer pipeline is driven by the
        # dense StaticPersistentTileScheduler exactly as the working dense
        # kernel does.
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, block_idx, cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        # Override the (m, n) tile coord with the grouped work-list values;
        # keep the scheduler-provided validity/structure.
        work_tile = WorkTileInfo(
            (
                moe_m_row // Int32(self.tile_shape_mnk[0]),
                moe_n_col // Int32(self.tile_shape_mnk[1]),
                Int32(0),
            ),
            work_tile.is_valid_tile,
        )

        # Pipeline states
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # Guard: skip invalid tiles (CTAs beyond the problem shape) to avoid
        # out-of-bounds gC_mnl access and barrier deadlock. Invalid CTAs still
        # hit the epilogue barrier below so the named barrier does not hang.
        if work_tile.is_valid_tile:
            # MMA warp group
            if warp_idx < self.num_mma_warps:
                cute.arch.setmaxregister_increase(self.mma_register_requirement)

                # Copy atoms for SMEM->RMEM
                atom_copy_ldmatrix_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.a_layout.is_m_major_a(), 4
                    ),
                    self.a_dtype,
                )
                atom_copy_ldmatrix_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.b_layout.is_n_major_b(), 4
                    ),
                    self.b_dtype,
                )
                smem_tiled_copy_A = cute.make_tiled_copy_A(
                    atom_copy_ldmatrix_A, tiled_mma
                )
                smem_tiled_copy_B = cute.make_tiled_copy_B(
                    atom_copy_ldmatrix_B, tiled_mma
                )

                atom_copy_ldmatrix_SF = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.sf_dtype,
                )
                smem_tiled_copy_SFA = cute.make_tiled_copy(
                    atom_copy_ldmatrix_SF,
                    self._get_layoutSFA_TV(tiled_mma),
                    (
                        cute.size(tiled_mma.permutation_mnk[0]),
                        cute.size(tiled_mma.permutation_mnk[2]),
                    ),
                )
                smem_tiled_copy_SFB = cute.make_tiled_copy(
                    atom_copy_ldmatrix_SF,
                    self._get_layoutSFB_TV(tiled_mma),
                    (
                        cute.size(tiled_mma.permutation_mnk[1]),
                        cute.size(tiled_mma.permutation_mnk[2]),
                    ),
                )

                thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
                thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
                tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
                tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
                tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
                tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

                thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
                thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
                tCsSFA_copy_view_full = thr_copy_ldmatrix_SFA.partition_S(sSFA)
                tCrSFA_copy_view_full = thr_copy_ldmatrix_SFA.retile(
                    tCrSFA_full
                )
                tCsSFB_copy_view_full = thr_copy_ldmatrix_SFB.partition_S(sSFB)
                tCrSFB_copy_view_full = thr_copy_ldmatrix_SFB.retile(
                    tCrSFB_full
                )

                self._run_grouped_consumer(
                    work_tile,
                    gC_mnl,
                    sC,
                    sSFA,
                    sSFB,
                    accumulators,
                    tiled_mma,
                    mma_atom,
                    tma_atom_c,
                    mainloop_pipeline,
                    mainloop_consumer_state,
                    k_tile_iter_cnt,
                    lambda x: x,
                    alpha_value,
                    tCsA_copy_view,
                    tCsB_copy_view,
                    tCsSFA_copy_view_full,
                    tCsSFB_copy_view_full,
                    tCrA_copy_view,
                    tCrB_copy_view,
                    tCrSFA_copy_view_full,
                    tCrSFB_copy_view_full,
                    tCrA,
                    tCrB,
                    tCrSFA_full,
                    tCrSFB_full,
                    smem_tiled_copy_A,
                    smem_tiled_copy_B,
                    smem_tiled_copy_SFA,
                    smem_tiled_copy_SFB,
                    tidx,
                    warp_idx,
                )

            elif warp_idx == self.tma_load_warp_id:
                cute.arch.setmaxregister_decrease(self.load_register_requirement)

                self._run_grouped_producer(
                    work_tile,
                    tma_atom_a,
                    tAgA,
                    tAsA,
                    tma_atom_sfa,
                    tAgSFA,
                    tAsSFA,
                    tma_atom_b,
                    tBgB,
                    tBsB,
                    tma_atom_sfb,
                    tBgSFB,
                    tBsSFB,
                    sSFA,
                    sA,
                    mSFA_mkl,
                    mA_mkl,
                    mSFB_nkl,
                    mB_nkl,
                    mainloop_pipeline,
                    mainloop_producer_state,
                    k_tile_iter_cnt,
                    k_tile_start,
                    tile_expert,
                    tile_m_row,
                    tile_n_col,
                    tile_idx,
                    gemm_n,
                )
                mainloop_pipeline.producer_tail(mainloop_producer_state)

            if cutlass.const_expr(self.enable_pdl):
                griddepcontrol_launch_dependents()


    _GROUPED_KERNELS: dict[int, _Qwen3xNvfp4MoeSm120Kernel] = {}
    _GROUPED_LOCK = threading.Lock()
    _GROUPED_BUFS: dict[int, dict[str, torch.Tensor]] = {}


        else:
            # Invalid CTA: skip mainloop, but still hit the epilogue barrier
            # so the named barrier does not deadlock.
            if warp_idx < self.num_mma_warps:
                self.epilog_sync_barrier.arrive_and_wait()
def _get_grouped_kernel(device_index: int) -> _Qwen3xNvfp4MoeSm120Kernel:
    kernel = _GROUPED_KERNELS.get(device_index)
    if kernel is None:
        with _GROUPED_LOCK:
            kernel = _GROUPED_KERNELS.get(device_index)
            if kernel is None:
                kernel = _Qwen3xNvfp4MoeSm120Kernel(device_index)
                _GROUPED_KERNELS[device_index] = kernel
    return kernel


def _get_grouped_buffers(device_index: int) -> dict[str, torch.Tensor]:
    """Per-device scratch for the grouped launcher (CUDA-graph safe: fixed
    capacity, reused across steps, no per-call allocation)."""
    bufs = _GROUPED_BUFS.get(device_index)
    if bufs is None:
        with _GROUPED_LOCK:
            bufs = _GROUPED_BUFS.get(device_index)
            if bufs is None:
                dev = torch.device("cuda", device_index)
                bufs = {
                    "w13_out": torch.empty(
                        _MAX_M, 2 * 512, dtype=torch.bfloat16, device=dev
                    ),
                    "x2": torch.empty(
                        _MAX_M, 512 // 2, dtype=torch.uint8, device=dev
                    ),
                    "sfa2": torch.empty(
                        128 * _ceil_div(_MAX_M, 128),
                        _ceil_div(512 // _SF_VEC_SIZE, 4) * 4,
                        dtype=torch.float8_e4m3fn,
                        device=dev,
                    ),
                    "w2_out": torch.empty(
                        _MAX_M, 2048, dtype=torch.bfloat16, device=dev
                    ),
                }
                _GROUPED_BUFS[device_index] = bufs
    return bufs


def _run_grouped_gemm(
    x_sorted: torch.Tensor,
    sfa: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alphas: torch.Tensor,
    work_list: MoeWorkList,
    out: torch.Tensor,
) -> None:
    """One grouped SM120 NVFP4 GEMM launch over the MoeWorkList.

    Args:
        x_sorted: uint8 [m_pad, K/2] packed fp4 activations in the sorted
            layout, padded to at least the highest sorted row referenced by
            the work list.
        sfa: fp8 swizzled activation scales covering the padded rows,
            [128*ceil(m_pad/128), 4*ceil((K/16)/4)].
        weight: uint8 [E, N, K/2] packed fp4 expert weights, k-contiguous.
        weight_scale: fp8 [E, n_pad, k_pad] swizzled weight block scales.
        alphas: fp32 [E] folded alpha = input_scale * weight global scale.
        work_list: MoeWorkList for this projection (built per step).
        out: bf16 [m_pad, N] output in the sorted layout (preallocated).
    """
    k = x_sorted.shape[1] * 2
    n = weight.shape[1]
    device_index = x_sorted.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    m_pad = x_sorted.shape[0]

    kernel_obj = _get_grouped_kernel(device_index)
    compiled = kernel_obj.compile((k, n), m_pad)
    sf_m = _ceil_div(m_pad, 128)
    sf_n = _ceil_div(weight.shape[0] * n, 128)
    sf_k = _ceil_div(k // _SF_VEC_SIZE, 4)
    compiled(
        x_sorted,
        weight.view(-1, weight.shape[-1]),
        out,
        sf_m,
        sf_n,
        sf_k,
        sfa.data_ptr(),
        weight_scale.data_ptr(),
        work_list.tile_expert,
        work_list.tile_m_row,
        work_list.tile_n_col,
        alphas,
    )


def _supports(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> bool:
    """Gate for the fast path; mirrors qwen3x_nvfp4_gemm._supports."""
    if hidden_states.device.type != "cuda" or hidden_states.ndim != 2:
        return False
    if hidden_states.dtype != torch.uint8 or topk_ids.ndim != 2:
        return False
    try:
        if torch.cuda.get_device_capability(hidden_states.device) != (12, 0):
            return False
    except RuntimeError:
        return False
    num_tokens, packed_k = hidden_states.shape
    if num_tokens * topk_ids.shape[1] > _MAX_M:
        return False
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        return False
    # w13: [E, 2*N, K/2] packed fp4; w2: [E, hidden, inter/2] packed fp4.
    kn1 = (packed_k * 2, w13_weight.shape[1])
    kn2 = (w2_weight.shape[2] * 2, w2_weight.shape[1])
    return kn1 in _SUPPORTED_GEMM_SHAPES and kn2 in _SUPPORTED_GEMM_SHAPES


# ---------------------------------------------------------------------------
# OPT-1: correctness-reference host loop over the proven dense SM120 kernel.
# ---------------------------------------------------------------------------


# e2m1 decode LUT: low/high nibbles index the 8 positive / 8 negative values.
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _dequant_nvfp4(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Dequant packed e2m1 (uint8 [..., K/2], k-contiguous) to fp32 [..., K]."""
    lut = _E2M1_LUT.to(packed.device)
    # The LUT holds the 8 magnitudes; the sign bit is handled separately
    # below. Masking it off keeps the gather indices in [0, 8) -- indexing
    # the 8-entry LUT with the full 4-bit nibble (0..15) trips the
    # device-side index-out-of-bounds assert on nibble values >= 8.
    lo = lut[(packed & 0x7).long()]
    hi = lut[((packed >> 4) & 0x7).long()]
    mag = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    sign_bits = torch.stack((packed & 0x8, packed & 0x80), dim=-1).reshape(
        *packed.shape[:-1], -1
    )
    signs = torch.where(sign_bits != 0, -1.0, 1.0)
    return (mag * signs)[..., :k]


def _unswizzle_blockscale(swizzled: torch.Tensor, m: int, k_groups: int) -> torch.Tensor:
    """Invert swizzle_blockscale for a [E, M_pad, K_pad] fp8 scale tensor.

    Returns fp32 [E, m, k_groups] in plain row-major block order.
    """
    if swizzled.dim() == 2:
        swizzled = swizzled.unsqueeze(0)
    e, m_pad, k_pad = swizzled.shape
    scale = swizzled.reshape(e, m_pad // 128, k_pad // 4, 32, 4, 4)
    scale = scale.permute(0, 1, 4, 3, 2, 5).reshape(e, m_pad, k_pad)
    return scale[:, :m, :k_groups].float()


def _run_dense_gemm(
    act: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_sf_swizzled: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """One dense SM120 NVFP4 GEMM via the proven kernel's public entry.

    Args:
        act: uint8 [m, K/2] packed fp4 activations (contiguous).
        act_sf: fp8 swizzled activation scales [128*sf_m, K/16] row-major.
        weight: uint8 [N, K/2] packed fp4 weights, k-contiguous (contiguous).
        weight_sf_swizzled: fp8 [N_pad, K_pad] swizzled weight scales.
        alpha: fp32 scalar tensor, the folded A2 = input_scale * weight_scale.
    """
    from sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm_sm120 import (
        _run_qwen3x_nvfp4_gemm,
    )

    return _run_qwen3x_nvfp4_gemm(
        act,
        weight.T,  # -> [K/2, N] column-major, as the dense path expects
        act_sf,
        weight_sf_swizzled.T,  # -> [K_pad, N_pad] column-major
        alpha.reshape(1),
    )


def _opt1_host_loop(
    hidden_states: torch.Tensor,
    hidden_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_alpha: torch.Tensor,
    *,
    activation: str,
) -> torch.Tensor:
    """OPT-1 host loop: dense SM120 NVFP4 GEMM per non-empty expert.

    The activation is gathered into the expert-sorted layout once (with its
    swizzled SFA), then per expert e over rows [off_e, off_{e+1}):
      gemm1: (M_e, K=2048) @ w13[e] -> (M_e, 1024), SwiGLU over the two
             512-wide halves in torch -> (M_e, 512)
      gemm2: (M_e, K=512) @ w2[e]  -> (M_e, 2048), router-weighted and
             scattered back into the [num_tokens, 2048] output.
    """
    num_tokens, packed_k = hidden_states.shape
    k1 = packed_k * 2
    n13 = w13_weight.shape[1]
    inter = n13 // 2
    hidden = w2_weight.shape[1]
    num_experts = w13_weight.shape[0]
    device = hidden_states.device

    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    top_k = topk_ids.shape[1]

    # Sort routed slots by expert id (stable: preserves (token, k) order
    # within an expert, which the router-weight gather relies on).
    sort_perm = torch.argsort(flat_ids, stable=True)
    sorted_experts = flat_ids[sort_perm]
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    offsets_cpu = offsets.cpu()
    counts_cpu = counts.cpu()
    src_tokens = torch.div(sort_perm, top_k, rounding_mode="floor").cpu()

    # Router weights in the sorted layout. Activations are gathered per
    # expert from the original token rows (src token = slot // top_k).
    w_sorted = (
        topk_weights.reshape(-1)[sort_perm].to(torch.float32).contiguous()
    )

    # hidden_sf: swizzled SFA [128*sf_m, K/16] covering the *unexpanded*
    # token rows. The swizzle interleaves token rows within each 128-row
    # atom (one swizzled view row holds only a 1/128 slice of a token's
    # scales across k-groups), so per-token rows cannot be copied out of
    # the swizzled buffer directly. Unswizzle to the linear [token,
    # k_group] layout, gather in token index space, and re-swizzle per
    # expert below.
    sf_k1 = hidden_sf.shape[1]
    hidden_sf_lin = _unswizzle_blockscale(hidden_sf, num_tokens, sf_k1)[
        0
    ].to(hidden_sf.dtype)

    out = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device=device)

    for expert in range(num_experts):
        rows = int(counts_cpu[expert])
        if rows == 0:
            continue
        # Sorted rows for this expert -> original token rows.
        lo = int(offsets_cpu[expert])
        token_idx = src_tokens[lo : lo + rows].to(device)

        x_e = hidden_states.index_select(0, token_idx).contiguous()

        # Build this expert's SFA for gemm1: gather the token rows from the
        # linear layout and re-swizzle (pads rows to the 128-row atom).
        sf_m_pad = _ceil_div(rows, 128) * 128
        sfa1 = _swizzle_sfa_rows(hidden_sf_lin[token_idx])
        assert sfa1.shape[0] == sf_m_pad

        # gemm1 (gate_up): (M_e, 1024) = x_e @ w13[e]
        y13 = _run_dense_gemm(
            x_e,
            sfa1,
            w13_weight[expert].contiguous(),
            w13_weight_scale[expert],
            w13_alpha[expert].to(torch.float32),
        )
        gate, up = y13[:, :inter].float(), y13[:, inter:].float()
        if activation == "silu":
            act_e = torch.nn.functional.silu(gate) * up
        else:  # pragma: no cover - gated MoE here is always silu
            raise NotImplementedError(f"activation {activation!r}")

        # OPT-1 requantizes the intermediate to NVFP4 in torch. This is the
        # one place the reference is *not* bit-faithful to the production
        # path (flashinfer quantizes with its own kernel); the reference
        # check below tolerates this via the torch-reference comparison.
        a2_scale = w2_alpha[expert].to(torch.float32)
        x2_e, sfa2 = _torch_nvfp4_quantize(act_e, a2_scale, k2=inter)

        # gemm2 (down): (M_e, 2048) = x2_e @ w2[e]
        y2 = _run_dense_gemm(
            x2_e,
            sfa2,
            w2_weight[expert].contiguous(),
            w2_weight_scale[expert],
            a2_scale,
        )

        # Router-weighted scatter back to token rows.
        out.index_add_(0, token_idx, (y2.float() * w_sorted[lo : lo + rows, None]).to(torch.bfloat16))

    return out


# ---------------------------------------------------------------------------
# OPT-2: single grouped launch per projection over the MoeWorkList.
# ---------------------------------------------------------------------------


def _opt2_grouped_launch(
    hidden_states: torch.Tensor,
    hidden_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_alpha: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """OPT-2: two grouped launches (gate_up then down) over the work list.

    The activation is gathered into the expert-sorted layout once (with its
    SFA re-swizzled per 128-row atom), the work list is built on the host
    (on-device build is a later refinement, noted for CUDA-graph safety),
    and each projection is a single persistent grouped-GEMM launch. SwiGLU
    and the router-weighted scatter stay in torch (OPT-3 fuses them).
    """
    num_tokens, packed_k = hidden_states.shape
    k1 = packed_k * 2
    n13 = w13_weight.shape[1]
    inter = n13 // 2
    hidden = w2_weight.shape[1]
    num_experts = w13_weight.shape[0]
    device = hidden_states.device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    top_k = topk_ids.shape[1]

    sort_perm = torch.argsort(flat_ids, stable=True)
    sorted_experts = flat_ids[sort_perm]
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    m_total = num_tokens * top_k
    m_pad = _ceil_div(m_total, 128) * 128

    offsets_cpu = offsets.cpu()
    counts_cpu = counts.cpu()
    src_tokens = torch.div(sort_perm, top_k, rounding_mode="floor").cpu()
    w_sorted = (
        topk_weights.reshape(-1)[sort_perm].to(torch.float32).contiguous()
    )

    bufs = _get_grouped_buffers(device_index)

    # Gather the fp4 activations into the sorted layout (pad rows zero).
    x_sorted = bufs.get("x1")
    if x_sorted is None or x_sorted.shape[0] < m_pad:
        x_sorted = torch.zeros(
            m_pad, packed_k, dtype=torch.uint8, device=device
        )
        bufs["x1"] = x_sorted
    else:
        x_sorted.zero_()
    x_sorted[:m_total] = hidden_states.index_select(0, src_tokens.to(device))

    # Re-swizzle the activation SFA into the sorted layout. The swizzle
    # interleaves token rows within each 128-row atom, so the gather is done
    # in the linear layout then re-swizzled.
    sf_k1 = hidden_sf.shape[1]
    hidden_sf_lin = _unswizzle_blockscale(hidden_sf, num_tokens, sf_k1)[
        0
    ].to(hidden_sf.dtype)
    sfa1_sorted = torch.zeros(
        m_pad, sf_k1, dtype=hidden_sf.dtype, device=device
    )
    sfa1_sorted[:m_total] = hidden_sf_lin.index_select(
        0, src_tokens.to(device)
    )
    sfa1 = _swizzle_sfa_rows(sfa1_sorted)

    wl1 = build_moe_work_list(offsets, n13)
    _run_grouped_gemm(
        x_sorted,
        sfa1,
        w13_weight,
        w13_weight_scale,
        w13_alpha.to(torch.float32),
        wl1,
        bufs["w13_out"][:m_pad],
    )

    y13 = bufs["w13_out"][:m_total].float()
    gate, up = y13[:, :inter], y13[:, inter:]
    act = torch.nn.functional.silu(gate) * up

    # Per-expert requant of the intermediate (different alpha per expert
    # region), built in the linear sorted layout so the single swizzle below
    # keeps the 128-row atom alignment of the grouped kernel's SFA.
    x2 = bufs["x2"][:m_pad].view(m_pad, inter // 2)
    x2.zero_()
    sf_k2 = _ceil_div(inter // _SF_VEC_SIZE, 4) * 4
    sfa2_lin = torch.zeros(
        m_pad, sf_k2, dtype=torch.float8_e4m3fn, device=device
    )
    for expert in range(num_experts):
        rows = int(counts_cpu[expert])
        if rows == 0:
            continue
        lo = int(offsets_cpu[expert])
        a2_scale = w2_alpha[expert].to(torch.float32)
        x2_e, sfa2_e = _torch_nvfp4_quantize(
            act[lo : lo + rows], a2_scale, k2=inter
        )
        x2[lo : lo + rows] = x2_e
        sf_lin_e = _unswizzle_blockscale(
            sfa2_e, rows, inter // _SF_VEC_SIZE
        )[0].to(sfa2_lin.dtype)
        sfa2_lin[lo : lo + rows, : inter // _SF_VEC_SIZE] = sf_lin_e
    sfa2 = _swizzle_sfa_rows(sfa2_lin)

    wl2 = build_moe_work_list(offsets, hidden)
    _run_grouped_gemm(
        x2,
        sfa2,
        w2_weight,
        w2_weight_scale,
        w2_alpha.to(torch.float32),
        wl2,
        bufs["w2_out"][:m_pad],
    )

    # Router-weighted scatter back to token rows.
    out = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    out.index_add_(
        0,
        src_tokens.to(device),
        (bufs["w2_out"][:m_total].float() * w_sorted[:, None]).to(
            torch.bfloat16
        ),
    )
    return out


def _torch_nvfp4_quantize(
    x: torch.Tensor, global_scale: torch.Tensor, k2: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch NVFP4 quantize for the OPT-1 gemm2 activation.

    Mirrors flashinfer fp4_quantize: per-16 block scales as e4m3, values
    scaled by 1/(global_scale * sf) then cast to e2m1. Returns the packed
    uint8 [m, K/2] values and the *swizzled* fp8 SFA [128*sf_m, K/16].
    """
    m, k = x.shape
    assert k == k2
    blocks = x.reshape(m, k // _SF_VEC_SIZE, _SF_VEC_SIZE)
    amax = blocks.abs().amax(dim=-1)
    # e4m3 max normal is 448; e2m1 max is 6. sf = amax / 6, in e4m3.
    sf = (amax / 6.0).to(torch.float8_e4m3fn).float()
    sf = torch.where(sf == 0, torch.ones_like(sf), sf)
    gs = float(global_scale.item()) if global_scale.numel() == 1 else None
    if gs is None:
        gs = float(global_scale.reshape(-1)[0].item())
    q = blocks / (sf[..., None] * gs)
    q = q.clamp(-6.0, 6.0).reshape(m, k)
    packed = _torch_fp4_pack(q)
    sfa_lin = sf.to(torch.float8_e4m3fn).reshape(m, k // _SF_VEC_SIZE)
    sfa = _swizzle_sfa_rows(sfa_lin)
    return packed, sfa


def _torch_fp4_pack(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even e2m1 pack: fp32 [m, K] -> uint8 [m, K/2]."""
    m, k = x.shape
    mag = x.abs()
    # e2m1 levels: 0, .5, 1, 1.5, 2, 3, 4, 6
    levels = _E2M1_LUT.to(x.device)
    idx = (mag.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    sign = (x < 0).long() << 3
    nib = (idx | sign).to(torch.uint8)
    lo, hi = nib[:, 0::2], nib[:, 1::2]
    return lo | (hi << 4)


def _swizzle_sfa_rows(sfa_lin: torch.Tensor) -> torch.Tensor:
    """Row-major [m, K/16] fp8 SFA -> swizzled [128*sf_m, K_pad] layout.

    Inverse of _unswizzle_blockscale for the activation (E=1) case.
    """
    m, sf_k = sfa_lin.shape
    m_pad = _ceil_div(m, 128) * 128
    sf_k_pad = _ceil_div(sf_k, 4) * 4
    padded = torch.zeros(m_pad, sf_k_pad, dtype=sfa_lin.dtype, device=sfa_lin.device)
    padded[:m, :sf_k] = sfa_lin
    s = padded.reshape(1, m_pad // 128, 4, 32, sf_k_pad // 4, 4)
    s = s.permute(0, 1, 4, 3, 2, 5).contiguous()
    return s.reshape(m_pad, sf_k_pad)


# ---------------------------------------------------------------------------
# OPT-1 reference check: torch dequant reference + optional flashinfer.
# ---------------------------------------------------------------------------


def _opt1_torch_reference(
    hidden_states: torch.Tensor,
    hidden_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_alpha: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch dequant MoE reference (fp32 matmuls, no kernel)."""
    num_tokens, packed_k = hidden_states.shape
    k1 = packed_k * 2
    n13 = w13_weight.shape[1]
    inter = n13 // 2
    hidden = w2_weight.shape[1]
    num_experts = w13_weight.shape[0]
    k2 = w2_weight.shape[2] * 2
    device = hidden_states.device

    # Dequantize activation once (values * sfa), without the input global
    # scale: that factor is folded into the per-expert alpha at the GEMM.
    x_dq = _dequant_nvfp4(hidden_states, k1)
    sfa = _unswizzle_blockscale(hidden_sf, num_tokens, k1 // _SF_VEC_SIZE)[0]
    x_dq = x_dq * sfa.repeat_interleave(_SF_VEC_SIZE, dim=1)

    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=device)
    for token in range(num_tokens):
        for j in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, j])
            w13 = _dequant_nvfp4(w13_weight[expert], k1)
            sfb13 = _unswizzle_blockscale(
                w13_weight_scale[expert], n13, k1 // _SF_VEC_SIZE
            )[0]
            w13_dq = w13 * sfb13.repeat_interleave(_SF_VEC_SIZE, dim=1)
            y13 = (x_dq[token] @ w13_dq.T) * float(w13_alpha[expert])
            gate, up = y13[:inter], y13[inter:]
            act = torch.nn.functional.silu(gate) * up

            w2 = _dequant_nvfp4(w2_weight[expert], k2)
            sfb2 = _unswizzle_blockscale(
                w2_weight_scale[expert], hidden, k2 // _SF_VEC_SIZE
            )[0]
            w2_dq = w2 * sfb2.repeat_interleave(_SF_VEC_SIZE, dim=1)
            # The down GEMM consumes the *requantized* intermediate; the
            # reference dequantizes the same torch-requantized values the
            # OPT-1 loop feeds its gemm2, so the two stay comparable.
            x2, x2_sf_sw = _torch_nvfp4_quantize(
                act.unsqueeze(0), w2_alpha[expert].float(), k2=inter
            )
            x2_dq = _dequant_nvfp4(x2, k2)
            x2_sf = _unswizzle_blockscale(x2_sf_sw, 1, inter // _SF_VEC_SIZE)[0]
            x2_dq = x2_dq * x2_sf.repeat_interleave(_SF_VEC_SIZE, dim=1)[0]
            y2 = (x2_dq @ w2_dq.T) * float(w2_alpha[expert])
            out[token] += float(topk_weights[token, j]) * y2.squeeze(0)
    return out


def _opt1_reference_check(
    num_tokens: int = 8,
    top_k: int = 8,
    num_experts: int = 256,
    hidden: int = 2048,
    inter: int = 512,
    *,
    device: str = "cuda",
    seed: int = 0,
    with_flashinfer: bool = True,
) -> dict:
    """Self-contained OPT-1 numerics check.

    Builds random NVFP4 activations + weights in the exact production
    layouts, runs the OPT-1 host loop, and reports max abs/rel error against
    (a) the pure-torch dequant reference and (b) flashinfer_cutlass fused MoE
    when importable. Returns a dict of error metrics.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    k1 = hidden
    k2 = inter

    def rand_fp4(*shape):
        # Random packed e2m1 nibbles.
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev)

    # Activations: random fp4 + swizzled SFA.
    hidden_states = rand_fp4(num_tokens, k1 // 2)
    sfa_lin = (torch.rand(num_tokens, k1 // _SF_VEC_SIZE, device=dev) * 0.5 + 0.5).to(
        torch.float8_e4m3fn
    )
    hidden_sf = _swizzle_sfa_rows(sfa_lin)

    # Weights + swizzled block scales in the production layout.
    w13_weight = rand_fp4(num_experts, 2 * inter, k1 // 2)
    w2_weight = rand_fp4(num_experts, hidden, k2 // 2)
    w13_sfb_lin = (torch.rand(num_experts, 2 * inter, k1 // _SF_VEC_SIZE, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    w2_sfb_lin = (torch.rand(num_experts, hidden, k2 // _SF_VEC_SIZE, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)

    def swizzle_e(s):
        e, n, kg = s.shape
        n_pad = _ceil_div(n, 128) * 128
        kg_pad = _ceil_div(kg, 4) * 4
        padded = torch.zeros(e, n_pad, kg_pad, dtype=s.dtype, device=s.device)
        padded[:, :n, :kg] = s
        t = padded.reshape(e, n_pad // 128, 4, 32, kg_pad // 4, 4)
        return t.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(e, n_pad, kg_pad)

    w13_weight_scale = swizzle_e(w13_sfb_lin)
    w2_weight_scale = swizzle_e(w2_sfb_lin)

    w13_alpha = (torch.rand(num_experts, device=dev) * 0.01 + 0.005).float()
    w2_alpha = (torch.rand(num_experts, device=dev) * 0.01 + 0.005).float()

    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=dev)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=dev), dim=-1
    ).float()

    got = _opt1_host_loop(
        hidden_states,
        hidden_sf,
        topk_ids,
        topk_weights,
        w13_weight,
        w13_weight_scale,
        w13_alpha,
        w2_weight,
        w2_weight_scale,
        w2_alpha,
        activation="silu",
    )

    ref = _opt1_torch_reference(
        hidden_states,
        hidden_sf,
        topk_ids,
        topk_weights,
        w13_weight,
        w13_weight_scale,
        w13_alpha,
        w2_weight,
        w2_weight_scale,
        w2_alpha,
    )

    def err(a, b):
        a = a.float()
        b = b.float()
        abs_err = (a - b).abs().max().item()
        denom = b.abs().max().clamp_min(1e-6).item()
        return abs_err, abs_err / denom

    result = {}
    abs_e, rel_e = err(got, ref)
    result["torch_reference"] = {"max_abs_err": abs_e, "max_rel_err": rel_e}

    if with_flashinfer:
        try:
            fi_out = _run_flashinfer_cutlass_reference(
                hidden_states,
                sfa_lin,
                topk_ids,
                topk_weights,
                w13_weight,
                w13_weight_scale,
                w13_alpha,
                w2_weight,
                w2_weight_scale,
                w2_alpha,
            )
            abs_e, rel_e = err(got, fi_out)
            result["flashinfer_cutlass"] = {
                "max_abs_err": abs_e,
                "max_rel_err": rel_e,
            }
        except Exception as exc:  # pragma: no cover - optional dep
            result["flashinfer_cutlass"] = {"error": repr(exc)}

    return result


def _run_flashinfer_cutlass_reference(
    hidden_states, sfa_lin, topk_ids, topk_weights,
    w13_weight, w13_weight_scale, w13_alpha,
    w2_weight, w2_weight_scale, w2_alpha,
):
    """Run flashinfer cutlass_fused_moe fp4 on the same inputs."""
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer import nvfp4_block_scale_interleave

    num_tokens = hidden_states.shape[0]
    hidden = w2_weight.shape[1]
    output = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=hidden_states.device)
    x_sf = nvfp4_block_scale_interleave(sfa_lin)
    quant_scales = [
        torch.ones(1, device=hidden_states.device),  # a1 global scale
        w13_weight_scale.view(torch.int32),
        w13_alpha,
        torch.ones(1, device=hidden_states.device),
        w2_weight_scale.view(torch.int32),
        w2_alpha,
    ]
    return cutlass_fused_moe(
        output=output,
        input=hidden_states,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=w13_weight.view(torch.long),
        fc2_expert_weights=w2_weight.view(torch.long),
        fc1_expert_biases=None,
        fc2_expert_biases=None,
        quant_scales=quant_scales,
        input_sf=x_sf,
    )


def _opt2_correctness_check(
    num_experts: int = 256,
    hidden: int = 2048,
    inter: int = 512,
    top_k: int = 8,
    *,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Compare the OPT-2 grouped kernel against the OPT-1 host-loop reference.

    Builds random NVFP4 MoE inputs at bs=1, 8, 32 and reports max abs/rel
    error of the grouped launch vs the validated dense-per-expert reference.
    """
    results = {}
    for bs in (1, 8, 32):
        torch.manual_seed(seed + bs)
        dev = torch.device(device)
        k1 = hidden

        def rand_fp4(*shape):
            return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev)

        hidden_states = rand_fp4(bs, k1 // 2)
        sfa_lin = (
            torch.rand(bs, k1 // _SF_VEC_SIZE, device=dev) * 0.5 + 0.5
        ).to(torch.float8_e4m3fn)
        hidden_sf = _swizzle_sfa_rows(sfa_lin)

        w13_weight = rand_fp4(num_experts, 2 * inter, k1 // 2)
        w2_weight = rand_fp4(num_experts, hidden, inter // 2)
        w13_sfb_lin = (
            torch.rand(num_experts, 2 * inter, k1 // _SF_VEC_SIZE, device=dev)
            * 0.5
            + 0.5
        ).to(torch.float8_e4m3fn)
        w2_sfb_lin = (
            torch.rand(num_experts, hidden, inter // _SF_VEC_SIZE, device=dev)
            * 0.5
            + 0.5
        ).to(torch.float8_e4m3fn)

        def swizzle_e(s):
            e, n, kg = s.shape
            n_pad = _ceil_div(n, 128) * 128
            kg_pad = _ceil_div(kg, 4) * 4
            padded = torch.zeros(
                e, n_pad, kg_pad, dtype=s.dtype, device=s.device
            )
            padded[:, :n, :kg] = s
            t = padded.reshape(e, n_pad // 128, 4, 32, kg_pad // 4, 4)
            return t.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(
                e, n_pad, kg_pad
            )

        w13_weight_scale = swizzle_e(w13_sfb_lin)
        w2_weight_scale = swizzle_e(w2_sfb_lin)

        w13_alpha = (
            torch.rand(num_experts, device=dev) * 0.01 + 0.005
        ).float()
        w2_alpha = (
            torch.rand(num_experts, device=dev) * 0.01 + 0.005
        ).float()

        topk_ids = torch.stack(
            [
                torch.randperm(num_experts, device=dev)[:top_k]
                for _ in range(bs)
            ]
        ).to(torch.int32)
        topk_weights = torch.softmax(
            torch.randn(bs, top_k, device=dev), dim=-1
        ).float()

        got = _opt2_grouped_launch(
            hidden_states,
            hidden_sf,
            topk_ids,
            topk_weights,
            w13_weight,
            w13_weight_scale,
            w13_alpha,
            w2_weight,
            w2_weight_scale,
            w2_alpha,
            activation="silu",
        )

        ref = _opt1_host_loop(
            hidden_states,
            hidden_sf,
            topk_ids,
            topk_weights,
            w13_weight,
            w13_weight_scale,
            w13_alpha,
            w2_weight,
            w2_weight_scale,
            w2_alpha,
            activation="silu",
        )
        abs_err = (got.float() - ref.float()).abs().max().item()
        denom = ref.float().abs().max().clamp_min(1e-6).item()
        results[bs] = {
            "max_abs_err": abs_err,
            "max_rel_err": abs_err / denom,
        }
    return results



def qwen3x_nvfp4_moe(
    hidden_states: torch.Tensor,
    hidden_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_alpha: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor | None:
    """Grouped NVFP4 MoE for Qwen3.6 decode shapes, or ``None`` for fallback.

    Args:
        hidden_states: packed NVFP4 activations, uint8 [num_tokens, hidden/2].
        hidden_sf: activation block scales, fp8 e4m3 in the SM120 quantize
            path layout (row-major, padded to 128 rows).
        topk_ids: int32 [num_tokens, top_k] expert ids.
        topk_weights: fp32/bf16 [num_tokens, top_k] router weights.
        w13_weight: packed fp4 gate_up weights, uint8 [E, 2*N, K/2], k-major.
        w13_weight_scale: fp8 e4m3 block scales in the swizzled CUTLASS
            layout (the existing ``w13_blockscale_swizzled`` param).
        w13_alpha: fp32 [E] folded alpha = input_scale * weight global scale
            (A2, prepared once at load like the dense fp4_gemm path).
        w2_weight / w2_weight_scale / w2_alpha: down-projection analogues.

    OPT-2 (current): single persistent grouped launch per projection over
    the MoeWorkList (one CTA per (expert, m_tile, n_tile) work item;
    SwiGLU / requantize / scatter stay in torch). Falls back to the OPT-1
    dense host loop when the grouped path cannot run.
    TODO(OPT-3): fuse SwiGLU into gemm1 epilogue; fold router weights and the
    bs*8 -> bs scatter-add into gemm2 epilogue.
    TODO(OPT-4): swap_ab for experts with <= 8 live rows (idle atom rows go
    from ~50% to ~0 at bs=1..8).
    """
    if not _supports(hidden_states, topk_ids, w13_weight, w2_weight):
        return None
    if activation != "silu":
        return None
    try:
        return _opt2_grouped_launch(
            hidden_states,
            hidden_sf,
            topk_ids,
            topk_weights,
            w13_weight,
            w13_weight_scale,
            w13_alpha,
            w2_weight,
            w2_weight_scale,
            w2_alpha,
            activation=activation,
        )
    except NotImplementedError:
        pass
    return _opt1_host_loop(
        hidden_states,
        hidden_sf,
        topk_ids,
        topk_weights,
        w13_weight,
        w13_weight_scale,
        w13_alpha,
        w2_weight,
        w2_weight_scale,
        w2_alpha,
        activation=activation,
    )


__all__ = [
    "MoeWorkList",
    "build_moe_work_list",
    "qwen3x_nvfp4_moe",
    "_opt1_reference_check",
    "_opt2_correctness_check",
]
