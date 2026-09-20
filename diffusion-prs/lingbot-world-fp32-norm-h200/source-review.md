# LingBot World causal normalization review

Reviewed final source: `89b3700dbc9146727cbb368b6859f5a18e79ae66`, based on main `dd83b54611897a5f80f4df59e69756bd2fb4b8ab`. Seven changed files, 770 insertions and 30 deletions. Earlier experimental-source reviews and failed attempts are preserved separately.

## Implementation and numerical contract

The native causal block normalizes in FP32, modulates in FP32, then stores BF16. Its later cross-attention LayerNorm also uses FP32 normalization and affine parameters before storing BF16. The patch reuses the existing 128-thread, vectorized-four ATen-compatible Welford reduction and adds two FP32 epilogues. Modulation keeps multiplication and addition separate (`enable_fp_fusion=False`); affine explicitly uses `tl.fma`, matching Torch's native vectorized affine epilogue. No intermediate BF16 rounding is inserted into these two chains.

The original BF16/FP8 branches remain the default. Current main's PR39983 adds `HAS_SHIFT` and a scale-only `shift=None` path; its full31-line KDA delta and90-line tests were read. The integration preserves the absence of a shift addition for `None`, including negative zero. The new FP32 mode is separate from that behavior.

The self-attention residual originally computed a normalized tensor that camera conditioning immediately invalidated. The new helper computes only the residual, preserving mixed BF16/FP32 arithmetic and omitting the positive-zero gate addition that would change signed zero. Camera modulation uses the existing rounded CUDA scale-shift kernel; it does not introduce a second runtime comparison gate around that established helper.

The three new mixed-dtype sites use per-signature live bit comparisons outside capture, then reuse the verified path. Unsupported layouts/dtypes/devices, gradients, compile, comparison failures, exceptions and unverified capture use native fallback. FP32 affine parameter conversion reuses the existing versioned cache. The complete causal-block test exercises both norm calls, residual and camera, asserts all three gates are enabled, and covers the cache-only early return. Ordinary LingBot blocks share only the camera optimization; their normalization call sites remain original.

## Source evidence

Read the installed Torch commit `cf30153c4c131c8164ee7798e5022d810682e2cb`: Welford update/combine, vectorized FP32 normalization/affine epilogue, and128-thread dispatch. Read the full existing KDA kernel, the relevant PR34008 implementation/dispatch sections and its body, and the native FP32LayerNorm parameter cache. PR34008's reduction and prior rounded modulation are existing work, not new claims.

The first block of each forward receives `patch_embedding(...).flatten(2).transpose(1,2)`. The candidate profile retains native norm1 at block indices0,40,80,120,160, consistent with the explicit contiguous-input guard;195 later norm1 calls and199 cross norms are fused. Existing QKV packing is unchanged. Camera-affine caching is an existing SP-only path: `_should_cache_cam_conditioner` requires sequence sharding and Ulysses world size>1, so it is not active in the measured1GPU run. The profile therefore still includes camera-affine GEMMs. The candidate does not change that cache policy.

## Current validation and measurement

Final source passed8new GPU tests+15subtests,22upstream scale-only/None/negative-zero/compile/graph tests,14existing BF16 norm tests,15FP8 norm/quant tests, and29runtime/import tests:88tests+15subtests. All7files passed pre-commit and Mint build/link checks passed.

The H200 whole-site marker at[1,4680,5120] reports norm1 325.440->47.328us, cross norm185.536->43.264us, camera102.688->53.504us, residual66.400->43.168us. At9360tokens the respective pairs are624.416->81.728,349.920->73.568,188.704->97.344,124.128->78.720us. Marker bandwidth is logical tensor traffic, not measured DRAM throughput.

Paired native profiles contain5complete causal forwards,200blocks and199camera/cross-norm sites. Four non-overlapping launch-attributed regions total3202->1232kernels,144.726153->38.255021ms. These include norm1's small entry-affine setup and are cumulative GPU times, not request speedups. The first slicing attempt also included unrelated GPU events in the CPU submission window; it is preserved as `unfiltered-cpu-window`. The corrected forward3 slice contains only kernels whose launch belongs to that forward (1583baseline,1188candidate).

Four short-CLI ABBA groups preserve all observations. Worker improvements are 2.290%, 2.527%, 2.539% and 2.613%, with identical 73.830078125 GiB peak. R1 includes a 6.06 s candidate client observation after a post-save delay, and r4 client improvement is only 1.336%. These groups do not establish a stable short-CLI client improvement; their negative qualification decision remains archived.

The initial native ten-chunk pair produced 117 raw-exact frames. Two subsequent fixed ten-chunk ABBA groups also preserved identical outputs, but the second group had only 0.854% worker improvement because of a slow first chunk. All initialization and transport/save costs remain included. The ten-chunk negative qualification decision is preserved.

Two predeclared thirty-chunk ABBA groups are the final qualifying workload: 357 frames (9 + 29 x 12), same native bounded causal cache, source and sampling settings. R1 worker 75.557500 -> 73.264000 s (3.03544%), saved client 76.702326 -> 74.414894 s (2.98222%); r2 worker 75.568500 -> 72.916500 s (3.50940%), saved client 76.717485 -> 74.073432 s (3.44648%). All eight raw RGB arrays, decoded frame hashes and MP4 bytes match. Both groups exceed the predeclared 1.5% worker and saved-client threshold. The opening PR claim uses the smaller saved-client improvement, 2.98%.

All valid final-source nine-frame lossless requests also remain byte-exact. High baseline/candidate outputs match each other; against lossless, decoded SSIM mean/min is 0.97989128/0.97729409 and PSNR mean/min is 42.93970/42.30205 dB. The 9-, 117- and 357-frame contact sheets were visually inspected. NCU collection/analysis and decoded media audits completed. Native BCG is explicitly disabled; standalone graph tests do not establish model BCG support. Streaming peak memory was not separately sampled.

## Historical review synthesis

The mandatory exact-path sweep scanned all 32,639 episodes and found zero matches because these kernel paths and model additions postdate the June corpus. Widening to multimodal_gen and legacy sgl-kernel paths matched 226 inline threads across 99 PRs (370 human comments). A separate complete conversation sweep matched 915 episodes/PRs (3,878 comments). The top-ranked excerpts from each sweep were read.

Recurring substantive concerns: preserve existing backend and fallback behavior, test CUDA graph capture independently of ordinary eager calls, exercise production tensor shapes rather than generic small examples, and provide native command/environment plus kernel benchmarks. PR14717 specifically requested kernel tests and benchmarks; PR19249 distinguished already fused modulation paths from the still-unfused path. PR26910 called out exact 4D affine layout assumptions. These precedents support inspecting actual LingBot call sites and reusing existing kernels instead of duplicating QKV/cache optimizations.


## Prior source evidence

Reviewed full diffs: PR30518 (rename), PR28760 (direct-current attention), PR36521 (static 4D scale-shift launch), and open PR34385 (SP variants). Reviewed the applicable cache/model/pipeline/config/stage sections of PR30040. PR34004 supplies the already merged rounded CUDA modulation kernel. The existing QKV projection is already packed. Camera scale/shift has an existing SP-only cache, inactive in the measured1GPU case. None is counted as new work. PR38044 concerns LingBot Video MoE routing, a different model family.


## Profiler interpretation corrections

The generic three-table triage is retained unedited. Its FlashAttnFwdSm90 row is attention; cuDNN fprop rows are convolutions; the CuTe ScaleResidualNorm row is normalization, not GEMM. Its FP8 nvjet replacement suggestion does not apply to this unquantized BF16 run. CPU submission overlaps device execution, so long Python scopes alone do not establish a CPU bottleneck. The explicit forward/correlation attribution finds no unrelated kernels inside the five selected forward device spans.



## Automated triage limits

The final three-table report's FlashAttnFwdSm90 row is attention, not GEMM. Its MoE activation/quantization suggestion is inapplicable to LingBot's dense unquantized BF16 MLP; the older unfiltered report's FP8 nvjet suggestion is also inapplicable. The suggested LLM KV-write path is not evidence that this causal-video cache's layouts and ownership are compatible. No overlap opportunity passed the reporting threshold. The actual kernel launch ownership and module source, not these generic suggestions, determine the patch.

## Open upstream review

Final sweep reviewed metadata/file lists and relevant descriptions for31921(optional TAEHV),33036(stage overlap),35359(compile trajectory gate),37357(VideoMoE EP); none changes these call sites. PR34385 remains at the already-reviewed23e079SP head. No competing normalization patch was identified. The exact docs-path corpus query found no matches; the full exact/legacy/conversation implementation sweeps are retained.

## NCU interpretation

The full/source/PM reports establish norm1 whole-chain338.496->41.024us(6->1kernels),DRAM821483008->108945920B;cross norm188.320->40.480us(3->1),475884800->91410688B. Fused32/38registers/thread andzero local-memory sectors. Source long-scoreboard samples concentrate at epilogueinputreloadline259(1582/976)andstatisticsloadline215(565/735). `_not_issued` subsets are kept separate. The candidate retains the reduction tree and removes FP32 materialization; no unmeasured reduction-algorithm claim is made. Full interpretation and one bounded next experiment are in the NCU report.

## Camera-cache policy cross-check

Read the complete current cache predicate/helpers, source-tensor/version invalidation and related unit tests. `git blame` identifies4ef081b9033a19ff81e1c8183e643c988231d4ec/PR27297 as the SP-only policy. Its body and full LingBot-model delta are archived; the relevant cache hunks were read. It reports4H200 camera/transport improvements and additional single-GPU CI OOM validation. That does not establish either safety or speed of enabling the retained cache on all1GPU shapes. The current PR keeps that separate memory/performance policy unchanged; it only fuses normalization, residual and camera post-ops.
