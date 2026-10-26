# Primary paired benchmark evidence

All 12 predeclared process blocks are present. The analysis verifies source/input hashes, clean checkouts, each saved benchmark PNG hash, every archived NVML sample's GPU UUID/process ownership, all quality gates and all repeated-short identities.

`raw/` contains original request/timing/stage/NVML/source metadata, server launch commands and numeric quality reports. `response-metrics.json` removes the repeated base64 PNG `data` field from the original response. Complete original responses, PNGs and server logs are also archived under the task's local/remote evidence roots; public comparison PNGs and SHA256 manifests are at this branch's root.

The measured cell is Base1024x1024,50steps,CFG3,batch1/concurrency1,short prompt. Other prompts check quality, not primary performance. Four-process block medians give main/PR1.014053x and Omni/PR1.022223x; all four pairs are positive. No claim about the full frozen matrix, other models, or BCG support.

Startup-to-ready has different framework-native boundaries, documented in the PR; reused compiler caches mean it is not a cold-compile comparison. Native Omni denoise instrumentation and matched final cuDNN Torch profiles are separate diagnostics.

The reproduction scripts preserve the actual `/task` container mount and task-owned host paths. Adapt these filesystem paths for another machine while retaining the model revision, source SHAs, launch flags, request contract, block order and warmup/repeat counts. Each arm must run serially on the same GPU, with no foreign compute process.
