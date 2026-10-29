"""Separate CPU/fallback diagnostic for the final dispatcher; no GPU timing."""
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch
from sglang.multimodal_gen.runtime.models.dits import sana


def main():
    out = Path('/task/outputs/diagnostics/sana-video-ln-final-cpu-compile1')
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1101)
    x = torch.randn(1, 37, 2240, dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 2240, dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    norm = torch.nn.LayerNorm(2240, eps=1e-6, elementwise_affine=False)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph.code)
        return graph.forward

    def candidate(x, scale, shift):
        return sana.sana_ln_modulate(norm, x, scale, shift, allow_eager=True)

    with patch.object(sana, 'fused_layernorm_modulate_raw', side_effect=AssertionError('CPU/compile must not dispatch fusion')):
        with torch.no_grad():
            expected = norm(x) * (1 + scale) + shift
            eager = candidate(x, scale, shift)
            compiled = torch.compile(candidate, backend=backend, fullgraph=True)
            compiled_out = compiled(x, scale, shift)
    assert torch.equal(eager, expected) and torch.equal(compiled_out, expected)
    assert len(graphs) == 1 and not sana._SANA_LN_MOD.verified_sigs
    assert not sana._SANA_LN_MOD.disabled and not sana._SANA_LN_MOD_DISABLED
    (out/'compiled-graph.py').write_text(graphs[0])
    result = dict(passed=True, device='cpu', shape=list(x.shape), dtype=str(x.dtype), eager_exact=True,
                  actual_torch_compile_graphs=len(graphs), compiled_backend='FX forward, fullgraph=True', compiled_exact=True,
                  fusion_not_called=True, gate_not_poisoned=True, torch=torch.__version__,
                  source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='CPU eager and actual TorchDynamo tracing fallback; no compiled GPU model performance claim')
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
