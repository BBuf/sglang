# FLUX.1-dev lossless BCG evidence on H200

Native BF16 FLUX.1-dev, 2 H200 TP2, 1024x1024,50steps,seed42,guidance3.5. Eight formal A/B/B/A samples are retained. Engine E2E4.315577494->4.177754073s(-3.193626%),denoise4.119539942->3.911356315s(-5.053565%). All eight formal50stepPNGs are byte-identical.

- benchmark.json contains exactconditions,sourcecommits,allrows,andtimingdefinition.
- perf-*.json are originalformalperfdumps.
- lossless-*.png are formal50stepimages.
- replay-correctness.json records all12final-head eager/BCG pairs acrossTP1/TP2; tp*-*.png are corresponding six-step, intentionally unfinished denoising outputs used for replay-state correctness only. Comparison is within each topology, not acrossTP1/TP2.
- profile-*-digest.json aggregate pairedfullstage2steptraces, including CUDA runtime anddriverAPI counts. Rawtraces remainremote; hashes andsizesidentifythem. ProfilingtimeisnotanE2Ebenchmark.
- weight-cleanup.jsonl proves task-ownedmodelweightsreturnedtozero.
- The helper scripts preserve exact measuredsource/revision and originally used task-ownedmachinepaths. The PRbody hasportableCLIcommands.

Production source 5e91bc39e7a421be979448661b1b5d93db35cb48; final3d51468a54b8fad047b1d679c0968ac2919613e7 adds onlytests/docs. Baseline482e9f257bb9d9ac57c2245c93768a96ec70edbe. Profile showsGPUkernelcountunchanged2847 andcumGPUtime144.160->147.284ms; speedupcomesfrom fewerhostlaunches. Existinggraph-awarecustomallreduceusespull+pushratherthanallpush. Peakreservedmemory19.47265625->20.70703125GiB. High+BCGrejectedbyexistingqualityfusionguard.
