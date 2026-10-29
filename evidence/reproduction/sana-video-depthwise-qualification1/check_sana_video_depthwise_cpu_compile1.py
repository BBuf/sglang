"""Exercise CPU eager and actual fullgraph tracing fallback, without GPU timing."""
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch
from sglang.multimodal_gen.runtime.models.dits import sana, sana_video

out = Path('/task/outputs/diagnostics/sana-video-depthwise-final-cpu-compile1')
out.mkdir(parents=True, exist_ok=False)
torch.manual_seed(1101)
module = sana_video.GLUMBTempConv(32, 3).bfloat16()
x = torch.randn(2, module.conv_depth.in_channels, 7, 11, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
graphs = []

def backend(graph, _inputs):
    graphs.append(graph.code)
    return graph.forward

rows = []
with patch.object(sana_video, 'fused_depthwise_conv3x3_bias_glu', side_effect=AssertionError('CPU/compile must retain reference')):
    with torch.no_grad():
        compiled = torch.compile(module._depthwise_glu, backend=backend, fullgraph=True)
        for index in range(2):
            if index:
                x.add_(0.125)
                module.conv_depth.bias.mul_(0.75)
            expected = sana.sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
            eager = module._depthwise_glu(x)
            actual = compiled(x)
            assert torch.equal(expected, eager) and torch.equal(expected, actual)
            rows.append(dict(index=index, eager_exact=True, compiled_exact=True))
assert len(graphs) == 1
assert not sana_video._SANA_VIDEO_DEPTHWISE_GLU.verified_sigs
assert not sana_video._SANA_VIDEO_DEPTHWISE_GLU.disabled
(out/'compiled-graph.py').write_text(graphs[0])
result = dict(passed=True, device='cpu', shape=list(x.shape), dtype=str(x.dtype), rows=rows,
              actual_torch_compile_graphs=len(graphs), compiled_backend='FX forward, fullgraph=True',
              fusion_not_called=True, gate_not_poisoned=True, torch=torch.__version__,
              source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='CPU tensors; CUDA visible for preexisting import-time device query. Actual Dynamo tracing; no compiled GPU performance claim.')
(out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
