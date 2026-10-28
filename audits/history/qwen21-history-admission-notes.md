> Historical preparation notes. Resumed 2026-10-28: verified weights and initial native CPU/GPU audits now complete; T2I numeric gate failed, edit passed. See CONTINUE.md and qwen21-current-source-resume2-comparison.json. No optimization chosen yet.

# Qwen-Image 2.1 admission and history notes

Source-only preparation while Joy final primary timings occupy GPU0. No Qwen model execution, download, performance result, candidate branch or optimization choice yet. Frozen matrix remains 81 entries / 176 workloads; Base50-step CFG1 text-to-image/edit and Turbo's native8-sigma CFG1 text-to-image are retained. CPU seed1101 and original frozen prompts/assets are unchanged.

## History lookup and actual source reads

The model-pr-history skill catalog has no Qwen-Image2.1 diffusion family. Its query tokenizes `Qwen-Image-2.1`, including generic digits, and returns unrelated autoregressive families; exact `rg` across the SGLang/vLLM corpus found no model match. These unrelated hits were not used as evidence. Fresh GitHub inventories are archived in `qwen21-{sglang,omni}-pr-discovery-20261027.json`; they are discovery only.

Actual full diffs, descriptions and returned discussion were read for:

- SGLang [#46101](https://github.com/sgl-project/sglang/pull/46101), merged738e4535 and present in frozenaf21: image-conditioned prefill retains packed QKV; subsequent steps project into the same target rows and omit copying prefix K/V. Text prefixes remain compact. Eligibility includes verified packed/pack gates, single sample, no SP or padded prefix. Tests check pointer ownership and empty-prefix calls across cached steps. Do not reimplement retained image-prefix storage. The reported additional memory must be measured here; H200/B300 numbers are not SM120 evidence.
- SGLang [#46709](https://github.com/sgl-project/sglang/pull/46709), open at inspection: adds `output_factory`/`pass_output` to `eager_on_graph`, allocates graph-pool outputs before ending capture, validates layout and weakly retains pool storage, and bypasses allocation/copies outside capture. Moves Qwen prefix-pad attention, common attention and H3 call sites to it; tests weak ownership, direct output identity, mismatches and changed-input replay. This overlaps generic graph-output management; do not duplicate it. It is not included in the frozen baseline and was not cherry-picked. The PR reports bounded microbenchmarks, not a model-level performance win.
- Omni [#9272](https://github.com/vllm-project/vllm-omni/pull/9272), merged27fd77b4 and present in frozen817d: before a new prefill, release every decode-graph entry whose request-owner weakref is dead, preserving live owners. A new request recaptures even a previously used full key. Step cleanup clears only request KV/geometry; subsequent prefill releases dead graph copies. Review explicitly rejected retaining dead entries by image layout because prompt length and batch also vary. Runtime tests must record actual capture/replay and per-request memory, including same-shape requests, rather than assuming warm graphs persist. Existing graph support must remain enabled in a qualified fast Omni comparator.

Exact PR heads/base SHAs and source file hashes are in `qwen21-history-admission-source-manifest.json`.

## Native contracts checked in the frozen trees

SGLang uses native transformer, Qwen3VL encoder and VAE components. The model-specific validation requires width/height multiples of32 and derives actual step count from sigmas. Pipeline autocast is disabled; VAE/text encoder configuration is BF16 and CPU generator is the default. Base and Turbo share the native pipeline. The conditioning stage constructs a raw model template, drops system-role tokens, collapses image slots, and obtains pre-final-norm hidden states. Reference VAE uses posterior mode; image resize, RGBA/white vision compositing, latent channel/stride layout and actual scheduler grids need cross-framework audits before timing.

The SGLang denoising stage sends prefill directly to the model and only cached steps to the graph runner. The existing Qwen prompt padder supports a single sample and SP1; it left-pads and stacks prefix K/V to a text bucket. The graph break excludes padded rows from attention; overlong prefixes return unchanged and may miss replay. This is existing BCG support, not a new result. Any later work must prove actual request replay, no stale prefix across changed prompts, and net E2E benefit.

The Omni recipe and current caller expose native default regional compilation plus decode CUDA graphs; `--enforce-eager` disables both, while `--no-enable-cuda-graph-decode` disables only graphs. Qualify the native default combination, retaining faster valid paths. Never select eager merely to make SGLang look faster. TP/SP/CFG/LoRA and prefix-cache paths have different graph eligibility. Recorded H200 recipe speed is not a result on this machine.

Qwen's model uses true CFG. Base primary remains1, with the existing CFG4 extension separate. Turbo uses checkpoint sample_sigmas and omits explicit steps; never compare the8-step checkpoint with the50-step Base as one workload. Three input contracts are now prospectively frozen in `inputs/qwen21-*-workloads.json`. Preserve decoded RGB and alpha separately in numeric/visual checks; a blanket `.convert("RGB")` would hide a native transparency regression.

## Next execution prerequisites

The prepared direct ModelScope downloader selects24runtimefiles totaling33,131,601,593bytes and checks the pinned HF revisiond26bb612. Mirror sizes/LFS hashes already agree; downloaded bytes remain unverified. Do not run bulk downloads during Joy primary timing. Base and Turbo are distinct checkpoints; only Base downloader is prepared.

After Joy releases the GPUs: acquire task-owned weights, inspect frozen checkpoint configs, audit processor/template/noise/sigma/precision semantics on both frameworks, qualify native outputs, and capture a baseline model profile. Only then select one independent optimization, refresh relevant PR diffs and read its current callers. Existing BCG, prefix-cache, host-sync, quantized SwiGLU and common graph-output work are not new opportunities merely because they appear in a profile.
