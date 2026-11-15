# Current native Nomas review, 2026-11-15

This task remains 6000-only. Earlier hardware figures in PR descriptions are historical clues, never current-machine results. New task PR count remains1.

Actually read source diffs and descriptions:

- [49270](https://github.com/sgl-project/sglang/pull/49270): group128 INT8 dequant kernel multiplies in FP32 then rounds toBF16, validates metadata and keeps fallback. Already merged; do not duplicate. It is irrelevant to the common BF16 lane because frozenOmni rejects Nomas quantization.
- [49308](https://github.com/sgl-project/sglang/pull/49308), BBuf: extracts the existing ordered mean-square reduction, uses per-signature BitExactFusionGate, retains compiled epilogue rounding and camerahead64fallback. Already merged; do not propose this as new6000work. Description explicitly excludes cross-framework quality failures from speedupaggregate.
- [49354](https://github.com/sgl-project/sglang/pull/49354), BBuf: VAE norm+MRoPE epilogue, BF16head128, rope64/32/32, SM100+, preserves ATen means and compiled FP32 multiply/FMAorder, actualParameter-backed tests and gate. Already merged and currentSM120eligible; profileactualactivation before proposing furtherfusion. Actual runtime/kernel/test diffs read, notjusttitle.
- [47686](https://github.com/sgl-project/sglang/pull/47686): camera chunk1 softmax-singlekey shortcut selectsframeV, preservesSPtokenoffset. Already merged. No newcamera fastpathclaim for equivalentcode.
- [47516](https://github.com/sgl-project/sglang/pull/47516): counts allocatorreserved-minusallocated asavailable KVbudget, rejectsSPindivisiblevideo tokens and exemptsflush. Already merged.

Full fetchedsource/metadata preserved in nomas-pr-source-diffs04.json and nomas-pr47516-source04.json. This note does not claim effective review comments were read for these additionalPRs; earlier cleanup-style.md holds the required actualcleanupPR/review study.

Native contract re-read at frozenSG0745 / Omniaa2968:

- Nomas SG nativepipeline is present. Matrixaliasname difference is notmissing support.
- NativeBF16,8steps,shift5,CFG1,CPU42,24FPS,704x1280,49/121/241frames. CommonSP1/2only; nativeOmni rejectsTP/CFG/DP/quant/LoRA and batchoutputs>1. Full-modelcompileunsupported; nativecompiledpointwise/FlexAttention staysenabled.
- SGdefault10latentsegments differs ordinaryOmni wholepromptchunk. Exact single segment set inrequest; bothpreparedcaptionsmustmatchbyte-for-byte.
- `inputs/nomas-native-workloads01.json` materializes21qualityrequests beforefirst6000output: short/long x3seeds x3frames plusnonfirstshortrepeats. StructuredJSONincludesactions-last; noexternalPE. Allfrozenmatrixcells retained.
- Rightpadding,textlen1280afterusercrop,hidden_states[-3],actualtokenids/masks/noise/sigmas andloadedtensors requireverification. OriginalfullQwen wrapper cannotbe assumedidentical; materializer comparesall338usedlanguage tensors.
- SGnativefinalofflinesegment avoidsunusedKVcommit andcapacity. Omni currentlyretainslastcommit. Discloseactualnativebehavior; do not artificially disablecompetitorfastpaths or inventextraSGwork.
- SingleGPUcapacity: stillunproven. Admit49before121/241, retainOOMs. NoGPUrun orNomasperformanceclaimyet.
