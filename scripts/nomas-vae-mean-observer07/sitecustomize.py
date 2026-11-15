"""Quality-only receipts for actual native VAE mean-square dispatch."""
import os
import runpy

if os.environ.get('NOMAS_AUDIT_ARM') == 'sgl':
    observer = runpy.run_path('/task/scripts/nomas-native-observer01/sitecustomize.py')
    from sglang.multimodal_gen.runtime.models.vaes import nomas
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.stages import NomasDecodingStage

    original = nomas.rmsnorm_mean_square_bitexact
    decode = NomasDecodingStage.forward
    calls = {}

    def observed_mean_square(x):
        key = (tuple(x.shape), tuple(x.stride()))
        calls[key] = calls.get(key, 0) + 1
        return original(x)

    def observed_decode(self, batch, server_args):
        calls.clear()
        result = decode(self, batch, server_args)
        gate = nomas._MEAN_SQUARE_FUSION
        observer['record']('vae_mean_square_fusion',
            calls=[dict(shape=k[0], stride=k[1], count=v) for k,v in calls.items()],
            gate_disabled=gate.disabled, verified=gate.verified,
            verified_signatures=[dict(shape=list(s[0]),stride=list(s[1]),device=str(s[2])) for s in gate.verified_sigs])
        return result

    nomas.rmsnorm_mean_square_bitexact = observed_mean_square
    NomasDecodingStage.forward = observed_decode
