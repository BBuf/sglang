# Cosmos3 Super Text2Image: initial profile and prior-art review

Diagnostic baseline dd83b54611897a5f80f4df59e69756bd2fb4b8ab, pinned checkpoint daf3d374804be4c512c2135568a7cb95d4341d79. Native eager BF16, 2 H200 TP2/SP1, 1024 square, 35 steps, CFG4, seed42, manual resident DiT/VAE. The first offline admission failed solely because the snapshot omitted seven README image assets; the complete same-revision snapshot resolved it. Failed logs retained.

Native admission worker4.381948279s, denoise4.311467766s, client4.77s, peak69.068359375GiB. Baseline PNG visually inspected. Actual BCG probe explicitly disabled; its saved eager fallback is not BCG evidence.

Full trace has64 native model forwards. The third complete model forward is isolated by CUDA launch correlation (not the CPU timestamp window alone):2133kernels,100.001342ms GPU span,56.105293ms union device activity. Parent/child module times overlap and must not be added.64 GEN layers each have33 kernels. Native local QKV shape is[1,1024,40,128]:32Q/4K/4V heads. Cross-attention source launches contiguous Q/K copies, QK norm, repeated FP32-cache-to-BF16 casts, split rounded RoPE mul/add/copies, and KV prefix concatenation.

The existing fused rounded QKNorm/RoPE/KV-pack kernel already implements the required BF16 normalization and rotary rounding, arbitrary supported token strides and prefix packing. Read its Python eligibility/dispatch and CUDA work distribution, normalization/rotary arithmetic, packing and launch sizing. Read native split/fused Cosmos wrappers and cross-attention, whole GEN residual carry, sample defaults and config metadata. Existing packed QKV, residual-add/RMSNorm, SwiGLU, and UND cache remain existing work. No new-kernel claim.

PR34932 adds the rounded T1 path and cache cast. PR36571 enables single-Hopper Nano while retaining the old dense Super regression guard. Their archived full diffs and bodies were reviewed. This checkpoint uses hidden_act=silu and hidden5120; it is distinct from the older dense Super regression. The proposed scope adds only Hopper TP2/SP1 hidden5120 SwiGLU to the existing policy, retaining compile and dense-MLP exclusions. Current open Cosmos PRs35350(candidate trajectories)and28021(VAE layout)do not replace this chain. Our separate PR40386 targets the2048-wide dense Edge shape, not this TP2 SwiGLU path.

The generic three-table report confirms split QK/RoPE-related work but its MoE activation/quantization suggestion is inapplicable: these are unquantized dense SwiGLU GEN/UND pathways. FlashAttnFwdSm90 is attention, not GEMM. Fused residual-add/RMSNorm is already present. Profile CPU/device gaps are diagnostic, not a promised recoverable E2E gain.

A fresh A/B will use current main80da4432d085ed4d6166ef643d9fd2b829dbb0c5; Cosmos model and stage implementation is unchanged from the diagnostic baseline. Prototype86c40bd29225f21f986aff26101f1c47f58695b8 changes only the dispatch guard and its tests. Final candidate1a7e397700ad6f6d7c1bb5af719411c9937bec1f keeps identical runtime and adds production-chain tests, benchmarks and documentation. Final audit results are appended below.


## Final code and evidence review

Read all five changed files against main80da. The runtime change passes hidden_size into the existing eligibility policy and permits only Hopper, T1, hidden5120, silu, TP2/SP1 and uncompiled GEN layers. Existing TP1/Blackwell handling and dense Super exclusion remain. No kernel code, attention backend, TP collectives, QKV projection or MLP arithmetic is changed. Whole-chain CUDA tests compare actual split/native wrappers with the existing rounded packed wrapper, including strided GQA QKV, FP32-stored BF16-rounded caches, nonempty and empty prefixes, batch2 and changed-input graph replay. KV inputs/prefixes remain unchanged; Q mutation is intentional and benchmark inputs are restored outside timers.

The full human review sweep scanned32639threads and matched3threads/6comments inPR24994 for these paths; all matches were read. Reviewer requests concerned removing redundant parallel wrappers, preserving backend compatibility and using USPAttention. The patch adds no parallel wrapper or backend override and reuses the existing cross-attention packed-KV/USPAttention dispatch. See review-corpus-final-paths.txt.

Final H200 checks:137modeltests+50subtests and5kerneltests pass; precommit/registration checks and Mint build/broken-links pass. Marker at1024/4096tokens:196.256->47.312us and311.488->56.576us eager;98.592->14.656us and285.248->41.200us standaloneCUDAgraph. Those graph timings are not model BCG evidence.

Paired final native3-step traces each contain8modelcalls. Third complete forward by launch correlation:2133->725total kernels; precise disjoint QK/RoPE+prefix-pack sites1472->64kernels,6.189829->0.747231ms cumulative GPU time. Rank0 only. The baseline profile contains large TP allreduce waiting variation, so total profiler time and apparent idle fraction are not treated as recoverable E2E speed. All modeled32Q/4KV/head128/1024tokens/prefix23shapes agree with the native source and tests.

Two initial fixed ABBA groups passed the worker/client>=1.5%criterion. Group1worker7.38477%/saved-client6.46872%; group2has baseline outliers (worker17.72769%,client30.18268%) and is not a headline. A third complete ABBA was scheduled to check replication, with every original row retained. Final-evidence.json is authoritative for that result. All initial22requestrecords retained: native asset-admission failure, exact saved generations, highEager checks and disabledBCG attempts. Actual before/after2x2imagegrid viewed; allfour originalPNGfiles identical. High does not silently turn this into approximate or BCG performance evidence.

The audit initially encountered a missing optional BCG field in the preserved offline-failure record, then the first replication launch raced an unfinished script transfer. Both failed logs and corrected retry scripts are retained. No timed row or output was edited. Candidate/base sources stay pinned throughout. Cleanup is performed only after final audit and image-hash verification.

Final third group completed:worker4.350231191->3.966876359s(-8.81229%),saved-client4.815->4.480s(-6.95742%). All26requestrecords retained,19validlossless outputs exact. Cleanup313files/30weights/132529363587bytes->zero.
