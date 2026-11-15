# Source/review evidence read before implementation

This is a task-specific summary, not permission to repeat the broad scope of the historical cleanup PRs. This goal requires one minimal, independent optimization per PR.

## #44690

https://github.com/sgl-project/sglang/pull/44690 — description read; named file patches retrieved through the paginated files API after the unified diff endpoint rejected >300 files. Read `sites/bitexact_gate.py`, `vaes/autoencoder_dc.py`, `test/unit/test_qwen_image21.py`.

- One `BitExactFusionGate.run` owns verification, failure disablement and fallback. First-sight verification cannot synchronize inside compilation or graph capture; verified kernels can replay. Reuse this lifecycle instead of inventing another gate.
- AutoencoderDC factors the duplicated weight-load routine, but preserves the distinction between first construction and a later load after `state_dict()` has already initialized the module. Cleanup must preserve actual state transitions.
- Qwen tests replace a local `_RecordingBlock` reimplementation with the real `QwenImage21TransformerBlock.forward` and a recording attention component. Exercise the production ownership bug; avoid tests that would still pass if production regressed.
- The description explicitly leaves launch-overhead-sensitive sites unwrapped where new Python closures would cost time. Reuse is subordinate to measured model latency.
- No substantive inline/human review appeared in the retrieved comment snapshot. Do not invent reviewer endorsement.

## #48366

https://github.com/sgl-project/sglang/pull/48366 — description read; named patches read for `bitexact_gate.py` and `test_qwen_output_bias_absorption.py`.

- A small per-signature `can_attempt(sig)` centralizes verified/disabled/unsafe-capture checks; `_can_verify_now()` includes XPU capture as well as CUDA. Preserve necessary platform behavior when simplifying a shared gate.
- Test registration must match both hardware and installed packages. DarkSharpness's [effective review follow-up](https://github.com/sgl-project/sglang/pull/48366#issuecomment-6724485907) identifies a real counterexample: adding the Qwen bias test to B300 base-c exposed missing `diffusers`, because that suite does not install diffusion extras. A test existing or being registered is not proof of valid CI coverage.
- The description rejects tautologies, tests of copied code, and always-skipped hardware gates; record actual executed cases and shared baseline failures.

## #49189

https://github.com/sgl-project/sglang/pull/49189 — description and named source diff read for Cosmos3 multiview sharding, Qwen ModelOpt scale validation, Qwen2.1 unused Q/K norm, LTX aliases/unused locals, and their affected tests.

- Import the identical existing `shard_sequence` helper and remove the duplicate tests with the duplicate implementation.
- Move identical ModelOpt FP8 scale predicates to one existing quantization owner; retain finite/positive/device/dtype guards and actual callers.
- Delete unused standalone Q/K norm wrapper, its gate/import, and the abandoned path together, rather than leaving an unreachable compatibility shim.
- Remove unused aliases and write-only locals only after checking callers, including string/registry references.
- The PR description explicitly defers same-line changes already owned by open PRs. This goal will record such overlaps before implementing.
- The retrieved comments contain CI automation but no substantive human code review; no review consensus is inferred.

## Rules for this goal

Use existing operators/gates and clear ownership; no unmeasured generalized abstraction. Keep numerical rounding and request/capture state constraints explicit. Test real operator/model behavior, non-first requests, graph replay with changed inputs, supported fallbacks and appropriate CI dependencies. Self-review every final diff and remove experiment branches/configuration/dead helpers. Compare cleanup-before/after on the same GPU/configuration, then validate the committed head against frozen main. Historical measurements above are not current 6000 results for this goal.

## Current source recheck on 2026-11-13

The latest #49189 description, affected diffusion source patches and comment timeline were fetched again and read. The Qwen ModelOpt predicates, Cosmos sharding reuse, dead Q/K norm deletion and LTX alias cleanup are present in frozen main. These are prior art, not new optimization opportunities. The retrieved timeline still has CI and author rerun comments, with no substantive human code-review endorsement.

#49171 (https://github.com/sgl-project/sglang/pull/49171) was read at description and source-diff level: preserve meaningful compatibility boundaries, prove deleted plumbing reaches all callers, and keep configuration for the actual versioned image even when current main does not read it. The FutureWarning change and its focused regression assert actual user-visible behavior. Removing a knob solely from a current-main search is insufficient for a pinned historical recipe.

All style references are historical source evidence; performance and correctness require fresh runs on this goal’s fixed GPU UUIDs.
