# H200 Hunyuan QKV/RoPE evidence

Baseline `c842d32b0ab45753178c5f6420662d7812766aca`; candidate
`b1c6e3aaa0b7b13ba0d2597d998e066d68e516e8`.

One NVIDIA H200, native SGLang backend, eager, lossless, resident components,
request warmup, fixed prompt/seed/checkpoint. Complete commands and environments
are under `runs/`; eight runs use two A-B-B-A groups. Each arm changes only the
Hunyuan QKV/RoPE kernel source. Output files, all 65 decoded frames, and both
same-source baseline repeats agree exactly. No audio stream is generated.
`model-evidence.json` includes every comparison and source/environment audit.

The six registered kernel cases reproduce 1.3288–1.3412x geometric speedup.
The actual 17-frame diagnostic layouts reproduce 1.3260–1.3292x. The full
65-frame layouts (34680 image tokens) reproduce 1.3478–1.3559x in three passes. The model
saved-request means change 69.155→68.985 s and 69.310→69.085 s, only 0.25% and
0.32%. These measurements do not establish an end-to-end speedup.

`media/comparison.png` and `media/comparison.gif` are rendered from the actual
native outputs. Original videos are `media/baseline.mp4` and
`media/candidate.mp4`; both SHA256 is
`7211a465d16b3ce5a2db3c60ca510d101fb87f75f070d6e7242fa8e0b5df0a33`.
The image and GIF are previews; correctness checks every decoded frame.

`kda-summary.json` records immutable Kimi/Codex provenance and the explicit
FP32 rotary-table compatibility repair made during integration. The
`kernel-benchmark` directory contains the frozen reproduction harness and
workloads. Private model endpoint credentials and optimizer conversations are
not included. The host-process judging setup does not claim container isolation.

Validation scripts retain original campaign paths; adjust paths/ports and obtain
the same public checkpoint to reproduce on another machine. Installed package
metadata in environment.json differs from the explicitly imported source;
request.json and source-files.json pin the actual implementation.
