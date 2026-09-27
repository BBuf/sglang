# Current integrated kernel evidence

Current source: `620955f604199168eff2abdc8f8c198061fe2cec`. All40 reported module identities match independently tested production bytes; per-family tested commits are retained below. This does not establish full-model validation of the combined source.

NCHW GroupNorm now uses the earlier in-budget constant-spatial candidate: approximately1.0064x, replacing the rejected1.17x implementation. MXFP8 has a documented installed-FlashInfer CUDA-reference limitation onH200; independent CPU-host reference checks pass. Neither family has a completed model-level speed claim.

H200; three fresh-process frozen suites per family, CUPTI cold-L2 timing. All frozen correctness rows pass. Ratios are baseline/candidate latency; existing-test exceptions and actual-model results remain separate.

| Family | Cases | Three suite geomeans | Captured native-shape geomeans | Worst case | Tested commit |
|---|---:|---|---|---:|---|
| groupnorm_channels_last | 18 | 1.3346× / 1.3298× / 1.3299× | — | 0.9857× | `65921d58` |
| h3_indexed_modulation | 12 | 1.2172× / 1.2176× / 1.2154× | — | 1.0000× | `65921d58` |
| hunyuan_qkv_rope_pack | 6 | 1.3452× / 1.3251× / 1.3390× | — | 0.9902× | `65921d58` |
| ltx2_ada_values | 8 | 2.4410× / 2.4374× / 2.4524× | — | 1.1434× | `65921d58` |
| rotate_half_rope | 8 | 1.2710× / 1.2970× / 1.2858× | — | 1.1006× | `65921d58` |
| wan_nearest_upsample | 9 | 1.1555× / 1.1595× / 1.1656× | — | 0.8988× | `65921d58` |
| wan_rmsnorm_silu | 15 | 1.1258× / 1.1257× / 1.1272× | 1.1603× / 1.1589× / 1.1595× | 0.9488× | `65921d58` |
| wan_scale_shift | 5 | 1.0814× / 1.0844× / 1.0817× | 1.4762× / 1.4767× / 1.4759× | 0.9834× | `61d241b9` |
| zimage_native_rmsnorm | 10 | 1.0856× / 1.0854× / 1.0885× | — | 1.0000× | `65921d58` |
| zimage_qk_rmsnorm | 8 | 1.4804× / 1.4772× / 1.4754× | 1.5694× / 1.5618× / 1.5650× | 1.2222× | `65921d58` |
| joint_qkv_cat | 6 | 1.1017× / 1.1065× / 1.1074× | — | 1.0413× | `2b4fe0ef` |
| sana_conv_post | 18 | 1.0421× / 1.0410× / 1.0407× | — | 0.9746× | `2f976a8f` |
| modulate_scale_shift | 31 | 1.4716× / 1.4772× / 1.4723× | 1.4036× / 1.4075× / 1.4047× | 1.0118× | `aea48eee` |
| residual_gate_add | 33 | 1.2670× / 1.2715× / 1.2689× | 1.2167× / 1.2186× / 1.2159× | 1.0013× | `6509f75c` |
| inplace_qknorm_rope | 30 | 1.1327× / 1.1333× / 1.1339× | 1.0651× / 1.0655× / 1.0662× | 1.0156× | `6cd9f731` |
| timestep_embedding | 15 | 2.3903× / 2.3974× / 2.3885× | 9.7416× / 9.7000× / 9.6444× | 0.9574× | `ed7b3a92` |
| sanawm_qk_inv_rms | 8 | 1.0296× / 1.0248× / 1.0242× | 1.1384× / 1.1354× / 1.1367× | 0.9583× | `eeef3e32` |
| group_limited_topk | 8 | 1.0383× / 1.0399× / 1.0396× | 1.0113× / 1.0119× / 1.0113× | 1.0086× | `6cd9f731` |
| qwen_qkv_epilogue | 7 | 1.1303× / 1.1402× / 1.1279× | 1.0757× / 1.0779× / 1.0768× | 1.0679× | `6cd9f731` |
| ernie_rmsnorm_modulate | 16 | 1.1739× / 1.1717× / 1.1727× | 1.1965× / 1.1948× / 1.1956× | 0.9768× | `61fdfd7b` |
| wan_norm_silu_post | 19 | 1.1673× / 1.1669× / 1.1656× | 1.2062× / 1.2048× / 1.2054× | 0.9276× | `6bb4fa5d` |
| causal_conv3d_cat_pad | 17 | 1.1587× / 1.1596× / 1.1608× | 1.2556× / 1.2554× / 1.2556× | 0.9849× | `a7289517` |
| norm_infer | 6 | 1.1022× / 1.1018× / 1.1059× | 1.1195× / 1.1199× / 1.1207× | 1.0067× | `7d5cde80` |
| helios_qk_rope | 6 | 1.1397× / 1.1377× / 1.1359× | 1.2950× / 1.2954× / 1.2954× | 0.9535× | `1e94dde6` |
| ulysses_qkv_pack | 11 | 1.1676× / 1.1647× / 1.1612× | 1.0072× / 1.0069× / 1.0071× | 1.0069× | `1730ffa1` |
| dup_up3d_add | 6 | 1.5753× / 1.5785× / 1.5940× | 1.7414× / 1.7425× / 1.7418× | 1.1630× | `cbd3380c` |
| glm_qk_head_layernorm | 3 | 2.2737× / 2.2747× / 2.2583× | 1.5891× / 1.5914× / 1.5840× | 1.5840× | `98ff2d4c` |
| layernorm_modulate | 16 | 1.3032× / 1.3044× / 1.3050× | 1.1588× / 1.1596× / 1.1578× | 1.0613× | `98ff2d4c` |
| layernorm_modulate_fp8 | 11 | 1.3000× / 1.3060× / 1.3062× | 1.2022× / 1.2064× / 1.2043× | 1.0636× | `98ff2d4c` |
| vdn_gather_linear_state | 3 | 26.3809× / 26.2043× / 26.0127× | 8.7930× / 8.7426× / 8.6985× | 8.6985× | `2ee1b7fd` |
| wan_causal_cache | 8 | 1.2053× / 1.2092× / 1.2114× | — | 1.0484× | `cbd3380c` |
| sanawm_bigdn | 6 | 2.1663× / 2.1710× / 2.1650× | 1.7559× / 1.7558× / 1.7539× | 1.6846× | `eeef3e32` |
| qknorm_complex_rope | 6 | 1.0809× / 1.0849× / 1.0848× | 1.1016× / 1.1017× / 1.1015× | 1.0349× | `da15e6fa` |
| qknorm_complex_rope_kv | 4 | 1.1516× / 1.1465× / 1.1393× | 1.0029× / 1.0032× / 1.0030× | 0.9982× | `da15e6fa` |
| groupnorm_nchw | 13 | 1.0064× / 1.0064× / 1.0065× | — | 0.9623× | `9436dcf3` |
| mxfp8_producers | 16 | 8.4337× / 8.0888× / 8.0380× | — | 2.1244× | `7adcfa61` |
| vdn_delta_factors | 3 | 1.0040× / 1.0044× / 1.0047× | 1.0097× / 1.0097× / 1.0096× | 1.0010× | `7a69cc73` |
| interleaved_rope_fp64 | 9 | 1.0409× / 1.0365× / 1.0401× | 1.5997× / 1.6052× / 1.5960× | 0.9266× | `61d241b9` |
| scaled_residual_add | 2 | 1.0142× / 1.0015× / 1.0133× | 1.0029× / 1.0030× / 1.0000× | 1.0000× | `61d241b9` |
| usp_merge_heads | 10 | 1.1037× / 1.1043× / 1.1042× | 1.1675× / 1.1660× / 1.1644× | 0.9844× | `6ce3fcbd` |


Native-shape columns use only frozen rows explicitly tagged from model captures; values are regenerated while captured layouts are preserved. A dash means this suite has no separately tagged native subset. None of these columns is full-model timing.
VDN gather at2ee1b7fd passes three frozen cases x3 (suite26.381/26.204/26.013x; native8.793/8.743/8.699x), nine unchanged VDN tests,48 external records and24 changed-input graph replays. Sealed-source stream/exit-hook and mixed-flag probes also pass. FP8 LayerNorm at98ff2d4c passes11 frozen rows x3 (1.3000/1.3060/1.3062x; native~1.204x),31 unchanged tests,29 extra shapes and58 graph replays; same-file ordinary LayerNorm and GLM QK suites were rerun x3. DupUp3D atcbd3380c passes6 frozen rows x3 (1.5753/1.5785/1.5940x; native~1.742x),16 unchanged tests and79 extra records, including factor-four dropped-frame cases, different layouts, selector boundaries and an actual above-2^31-element index case. Same-file causal-cache suite reran x3. The prior rejected Dup candidate's results remain historical only.
Helios Q/K RoPE passes six frozen rows x3 (1.1397/1.1377/1.1359x; captured native shapes approximately1.295x),13 unchanged tests,103 external contract records and80 changed-input graph replays. Worst small row0.9535x is retained above. Ulysses QKV packing passes11 frozen rows x3, three unchanged tests,120 extra dtype/shape/world-size/stride/output-alias cases and128 changed-input graph replays. Their device implementations come from sealed Kimi submissions; final combined native model timing remains pending.
norm_infer at 7d5cde80 passes six frozen rows in three independent processes (1.1022/1.1018/1.1059x; captured native layouts approximately1.120x), 150 unchanged tests plus four subtests, and 78 additional dtype/partial-width/affine-flag/strided-input/explicit-output/selector-boundary cases. Thirty cases include changed-input CUDA Graph replay on nondefault streams;48 further special-value/in-place cases also pass. Its production bytes remain unchanged in the current tree; the VDN temporal candidate from7d5cde80 was removed after extended exactness checks failed, despite its frozen-suite and existing-test passes. It is not counted as an integrated family.
Causal CUDA cat/pad at a7289517 passes seventeen frozen rows in each of three independent processes (1.1587/1.1596/1.1608x; native-layout geomean ~1.2555x), eleven unchanged layout/model tests and seven external long-depth/large-grid/cropping/stream/graph/invalid-padding/alignment contracts. The new structured host entry restores the exact original RuntimeChecks; sealed Kimi device/launch statements are unchanged.
Wan norm-SiLU post at 6bb4fa5d passes nineteen rows x3, three unchanged existing tests with sixteen subtests, and seven external cases including both layouts above the 2^31 element-index boundary. Neither entry is a model-level speed claim.
SANA includes the unchanged baseline GLU operation; only bias+SiLU is modified. The CSV retains every case and repeat. Values below 1 are regressions and must remain visible in the eventual PR.

Existing tests: 379 selected tests passed at the earlier 12-family commit. Joy and SANA subsequently passed their unmodified relevant files. Residual passed 35 selected tests at 6509f75c. Flat modulation passed 12 unchanged tests at aea48eee; its large-index and changed-input stream/graph contracts are recorded separately. LayerNorm prefetch passed 15 LayerNorm/FP8 and 13 model-fast-path tests at 783b0cea, with partial-chunk and row-boundary contracts recorded separately. QKNorm/RoPE passed 1258 unchanged kernel tests across inplace, out-of-place and KV pack, plus 3 model-callsite tests at 4f8b9acf. Counts are not summed because selections can overlap.
Timestep embedding passed 792 unchanged tests plus 23 external dtype/guard/graph cases at ed7b3a92. SANA-WM inverse-RMS has no direct pre-existing test in test/; its eight frozen rows, declared guard, and six external dtype/offset/nondefault-stream/changed-input graph cases passed at f0ad9480. No repository tests were added or changed.
Historical LayerNorm 0fe637f1 at 61fdfd7 passed 28 unchanged LayerNorm/FP8/model-fast-path tests plus 38 external partial-width/row-boundary/changed-input graph cases. At 49c58e3a the same module also contains SM90 GLM QK launch tuning: all LayerNorm-modulation functions are AST-identical, and its sixteen frozen rows were rerun in three independent processes (1.2912/1.2935/1.2912x).
GLM QK head LayerNorm at 49c58e3a passes three frozen rows in each of three fresh processes (2.2307/2.2457/2.2444x); the captured native layout is 1.5933/1.5897/1.5905x. Thirty additional partial-head/255-256-257/4095-4096-4097-row, nondefault-stream and changed-input CUDA Graph cases pass. Thirty-one unchanged tests cover QK model fast paths plus shared LayerNorm/FP8 paths. This is an operator claim, not a model E2E claim.
Qwen QKV v15 and shared QKNorm v14 use independent C++ namespaces for the baseline/candidate comparison: the earlier Qwen v13 measurement was contaminated by GNU-unique occupancy-cache symbol interposition and is superseded. Only namespaces change in the timing harness; production kernel statements are unchanged. Qwen passed 14 existing and 14 external contract cases; TopK passed 28 existing and 24 NaN/Inf/tie/graph cases.
ERNIE RMSNorm modulation passes sixteen frozen rows in each of three processes and twelve external stream/offset/511-512-513-row/changed-input graph contracts. Existing model-fast-path tests fail 3/4 in BOTH baseline and candidate with the current RMSNorm backend, also when FLASHINFER_USE_CUDA_NORM=0. This is not a claim that the model fast path is enabled or its tests pass; native dispatch and final model validation remain required.

Flat modulation retains the original public data_ptr-based guard: public fullgraph is unsupported in both baseline and candidate on this Torch build; the existing registered custom-op boundary compiles and gives exact results.

Native combined A/B timings, dispatch proof against the final source, media comparison and publication remain separate pending requirements.

BiGDN at eeef3e32 passes six frozen rows x3 (suite2.166327/2.170968/2.165023x; native1.755901/1.755793/1.753869x), nine unchanged SANA tests,30 production-facade records including9 changed-input graph replays, two-GPU cache reuse,16 concurrent host calls and exit-only LazyDict callbacks. Original shared camera/state/Phase-C helpers remain byte-identical; the private implementation is used only by the stateless facade. Python dependencies are hash-qualified. Same-file inverse-RMS reran eight rows x3. SANA-WM and Z-Image full-model results are now separately recorded in model-benchmarks-current.md; remaining model lanes and publication-source checks are pending.

Qwen complex-RoPE at da15e6fa passes six plain and four KV frozen rows in each of three fresh processes,35 unchanged existing tests,290 expanded production records including32 changed-input CUDA Graph comparisons and two fullgraph compile comparisons. Plain captured-native geomean is about1.102x; KV captured-native geomean is only1.003x despite a1.14-1.15x suite mean. Original four-row shared helper,public signatures,guards and custom-op/fake boundaries remain unchanged. This is not a Qwen model E2E claim.


groupnorm_nchw: Original-budget constant-spatial candidate replaces rejected fused-finalize implementation. 55 external records including actual Hunyuan failure tensor and 9 changed-input graphs pass; 15 unchanged tests pass. Both full Hunyuan lossless/high lanes now pass eight-file exact-output audits at9436dcf3; timings are reported separately.

mxfp8_producers: 132 expanded production records plus8 targeted regressions pass. The original CUDA FlashInfer tests fail4/5 in both arms onH200 because installed0.6.18 compiles an empty SM90 kernel body; two sentinel probes show zero bytes written. Four independent CPU-host quantization oracle cases pass for baseline and candidate. Original test files unchanged; no model E2E gain claimed.


vdn_delta_factors: Original-budget616029bbe56b453c;13 unchanged tests,27 sealed-source records and38 production stream/Graph/two-device/fullgraph/concurrent-host records pass. Suite gain is about0.4%, captured native about1%; no model speed claim.

interleaved_rope_fp64: Earlier original-budgetc6d92a482f454c55 uses ordinary stream serialization. Later109d44 PDL version is rejected for producer-read races. Selected source passes64 expanded records and the adversarial producer check; production passes5 unchanged tests. Small-shape regressions remain visible.

scaled_residual_add: Earlier original-budget8a9421acc6f84185 uses ordinary Triton dispatch and a mask-free exact-block specialization. Later raw-cache version is rejected for offset-one misaligned loads.74 expanded records and1018 unchanged modulation tests pass. Captured native gain is only0%-0.3%.

wan_scale_shift: Same-file suite rerun after scaled-residual integration; its production functions remain AST-identical.


usp_merge_heads: Original-budget2235dd5d43364552, exact Kimi90358cb0. Ten frozen rows x3, ten unchanged tests,83 expanded contracts including36 changed-input Graph replays and actual >2^31 indices, plus22 two-device/concurrent-host records pass. Original task judge was infrastructure-blocked; independent replay and production evidence are retained. Native about1.166x; worst small shape0.984375x. Final affected multi-GPU model validation pending.
