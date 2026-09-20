# CI follow-up on 2026-09-21

All seven branches merge main80da4432d085ed4d6166ef643d9fd2b829dbb0c5, including the upstream #40411 scheduler fixtures and #40427 offline DFlash test fixes. No checkpoint was re-downloaded for this CI follow-up. Original model E2E/media evidence stays pinned to its measured revision; no new request-performance result is claimed for these merged heads. Source deltas limited to the PR files are attached.

| PR | Current-head targeted H200 validation |
|---|---|
|40374|97 kernel/model tests passed|
|40378|4 tests + 8 subtests passed|
|40384|6 kernel,46 config/padding + 7 subtests,7 model fast-path tests passed|
|40386|145 tests + 46 subtests passed|
|40388|5 eager-convolution tests passed|
|40405|7 kernel/site tests + 16 subtests passed|
|40425|8 LingBot tests + 15 subtests,37 shared-kernel compatibility tests,29 runtime/import tests passed|

The common upstream CPU fixture tests passed49tests+57subtests on the merged LingBot head; those files are identical across all seven branches. Initial local collection lacked parameterized; after installing parameterized0.9.0 the unchanged test files passed. Both attempts are retained. All PR files pass pre-commit.

LingBot's real B200 failure was the unconditional raw FP32 norm comparison for[2,63,2240],random. The helper explicitly requires live Torch-dispatch verification and the production site already rejects mismatches. The revised test executes that actual site for every shape/value/epilogue on all NVIDIA GPUs, checks exact returned bits and gate accept/disable state against the raw comparison, and retains strict raw equality on the validated Hopper architecture. No tolerance is weakened and no production path is changed. H200 validation passes; B200 validation awaits the new upstream CI run.

SenseNova's prior2GPU lane generated exact Wan videos but ran far above the stored latency baseline before the45-minute cancellation. This is a performance/coverage failure; the log does not establish its root cause. Assertions remain unchanged. The main merge also includes an independent upstream SenseNova shared-RMSNorm dispatch change, so original E2E timings must not be represented as measurements of the new head.
