# Nomas VAE121-frame dispatch audit

Frozen native decoder.decode uses window_starts(length,10,5), one DMD decode step and36decoder layers.121frames gives31latents and window starts[0,5,10,15,20,21]. Each10-latent window at1280x704 has361mean-square calls as observed on both49frame windows: per36layers8splitQK reductions plus2tokenRMS reductions, plusfinalRMS1. Expected2166calls/request, widths64/4096. This is a predeclared assertion for the actual observer, not a runtime success claim. Candidate13 must match all7real native main121MP4 hashes; no timings admitted fromobserver.
