# B300 diffusion PR visual artifacts

These artifacts were generated on NVIDIA B300 from SGLang main
`dbebc1deb42b00befa3d0de67265d7003994c1ad` and the corresponding PR trees.
Each before/after pair uses the same model configuration, prompt and seed.

- `flux2/`: FLUX.2 Klein Base, 1024x1024, seed 42, 50 steps
- `hunyuan/`: HunyuanVideo, 848x480x65 frames, seed 42, 30 steps
- `ernie/`: ERNIE-Image Turbo, 1024x1024, seed 42

Model weight caches were removed after each model's validation completed.
