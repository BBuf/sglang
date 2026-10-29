# Cleanup evidence reviewed before implementation

Retrieved live source patches, PR descriptions and available review discussions on 2026-10-26. These are style/prior-art references, not performance results for this goal. Selected source diffs were actually read; the full broad PRs have not been exhaustively audited.

## #39293 — obsolete diffusion plumbing

https://github.com/sgl-project/sglang/pull/39293

Read source patches for `runtime/distributed/parallel_state.py`, `runtime/managers/gpu_worker.py`, and `runtime/managers/scheduler.py`, plus the full description. Deleted unused pipe arguments and scheduler methods rather than retaining compatibility wrappers; reused `GroupCoordinator` rather than a redundant subclass. Deletion evidence includes callers/imports/registration/dynamic dispatch. Tests focus on real process-group lifetime and readiness/rank mapping. Description explicitly makes no model throughput claim without generation measurements.

Apply here: remove experiment plumbing from the final patch; preserve process/request/graph lifetime ownership and directly validate any changed lifetime. Avoid source-text/identity tests.

## #42391 — launcher validation and dispatch

https://github.com/sgl-project/sglang/pull/42391

Read the residual-gate JIT source patch and the diffusion kernel README patch; read the motivation, modifications, validation and performance discussion. Retain only the measured contiguous CUDA and transposed Triton implementations. Remove dead variants and blanket exception-to-disable caches. Backend choice depends on actual input device, numerical policy and genuine specialization; tensor legality belongs in the launcher. Preserve original dimensions for validation. Preserve first-use numerical checks when exactness depends on library reduction behavior.

Apply here: no tensor-scanning predicate duplicated at call site and launcher; no broad silent fallback masking errors. Existing numerical contracts remain explicit. Cleanup performance must include actual model call sites, with final SHA and library stack matched. The PR's own later model diagnostic did not exclude a small regression despite earlier microbench success, reinforcing this goal's cleanup-after model retest requirement.

## #40265 — numerical test consolidation

https://github.com/sgl-project/sglang/pull/40265

Read the full description and source patches for SANA-WM realtime/streaming fixtures, `test_layernorm_cutedsl_dispatch.py`, and `test_modulate.py`. Reused the parent fixture, parameterized duplicate numerical bodies without dropping the shape/dtype matrix, and kept independent numerical references and graph/fallback coverage. The description distinguishes skipped tests from executed checks and does not call cleanup an inference optimization.

Apply here: extend an existing parametrized numerical/lifecycle test for a real regression; avoid copying complete model implementations into test oracles, duplicated wrappers, and standalone single-shot timing inside correctness tests. Preserve tolerances, CI registration and case coverage.

## Review availability

The live review-thread endpoint returned no inline threads for #42391, #44690 and #40265. #42391's fetched timeline contained CI/control discussion and no additional substantive source-review instruction. #39293's fetched discussion consisted of CI trigger comments. No reviewer opinion is inferred from absent review text. #44690 was retrieved as a recent discovery reference, but its very large full patch is not claimed as manually audited.

## Existing optimization conflicts to inspect before choosing a patch

Live open-PR search is archived in `open-pr-search.json`. In particular do not duplicate Wan cross-attention KV caching (#46246), Wan residual/norm fusion (#46245), offload buffer reuse (#46438), video uint8-before-host-copy (#46439), conditioning hash parallelization (#46100), short Wan text encoding (#46328), FLUX.2 encoder layer pruning (#46324), quantization overhead fixes (#46303), CuTe norm row tiling (#46227), or the existing BCG/residency work. Search by exact target model/function again when a profile selects the optimization.
