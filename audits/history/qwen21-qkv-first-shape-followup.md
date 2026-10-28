# Unselected native QKV gate follow-up

The final-head compatibility diagnostics warm up512 before1024. Both frozenmain and candidate log `packed Q/K/V projection ... is not bit-exact ... falling back to eager` on the first512 warmup. The process-global BitExactFusionGate then keepsQKVseparate at1024 too. The actual1024 profile has224largeGEMMs rather than160:64extraQKV projections, and480->448total kernels for the MLP change. Nativeprimary1024 runs must use fresh1024-onlywarmup, as the earlier1024ABBA did, to retain its admitted nativeQKVfastpath. Preservecombinedshapecompatibility evidence but do notcallitrepresentative1024primaryprofile.

A possible independent follow-up is nativehigh-modeQKVpacking at512 or shape-awareverification, onlyaftercurrentmain/openPR/source review and actualnumerical/quality/performance validation. Do notimplement,count,orbundle this into activeMLPPR. Noquantitativewin inferredfromkernelcount alone.
