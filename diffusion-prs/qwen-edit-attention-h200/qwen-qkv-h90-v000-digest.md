### NCU Report Digest: Qwen QKV epilogue @38ee93fc36

Environment: ion8H200/SM90, driver595.71.05, PyTorch2.11.0+cu130, NCU2025.3.1. Actual shape image8152/text1365,24heads,128head-dim,BF16,randomizedBF16 norm weights. One warmed exact launch after5warmups; GPU0idle/0MiB beforeentry. Command and sourcecommit are inmeta.json. Full sections plusPMsampling collected45passes; clocks/cache left unchanged and NCU warns clocks were not fixed. Kernel replay measurements are diagnostic, not full-modeltiming. No comparable parentNCU exists yet. Standalone fullmodelABBA38ee vs482 is33.30735→33.06309s (-0.733%),all4PNGexact.

Headline: mixed memory latency and instruction overhead; medium-high confidence. Duration121.25us, DRAM61.72%/2.97TB/s,L2 70.55%,SM62.45%,achievedoccupancy89.35%,32registers/thread,zero spills,branch efficiency100%. Longscoreboard12.5cycles/instruction or56.67% ofwarpcycles;34.60% scheduler cycles have noeligiblewarp. This does not justify an arithmetic or occupancy rewrite.

Source evidence: no CUDA lineinfo orPTX was available (source.csv/kernel.ptx contain explicitwarnings), so the exportedSASS was inspected directly. Four32-bit LDG.E.CONSTANT cache loads at0x7562517cd0a0..d0d0 each have456816warps and1,827,264 excess theoreticalsectors. Their aggregate7,309,056matches all excessive sectors reported byNCU (25% of29,236,224). Eachlane accesses cache indices2*lane and2*lane+1 in separateinstructions, wasting half ofeach32-byte sector perinstruction; consecutivefloat2 loads can cover the samebytes with two64-bit loads. This mechanism is source/SASSinference with exactsectorcount agreement. The V-copy STG has a large long-scoreboard sample count, but changing that path is deliberately not part ofthis edit. PmSampling sections collected6passgroups at1.5us maxinterval and remainembedded inthe report; CLI has no samplingpage. Aggregate samples are not interpreted as a precise cross-kernel timeline.

Next Concrete Edit
- File: python/sglang/kernels/jit/csrc/diffusion/qwen_qkv_epilogue.cuh
- Change: replace thefour scalar cos/sin cache reads with two alignedfloat2 reads, keeping every normalization/RoPE arithmetic expression andBF16 store unchanged.
- Validation: actualshape andmixedstrides bitexacttests; productionmicro; samefocusedNCU command; requireSASS twoLDG64 cacheloads and reducedexcesssectors before any fullmodelclaim.
- Expected metric movement: cache-load instructioncount4→2 and modeledexcesssectors7,309,056→0; totalruntime may improve less because Vcopies andothermemorytraffic remain.
