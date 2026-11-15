"""CPU/meta audit of a key-only canonical-to-native checkpoint adaptation."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from sglang.multimodal_gen.configs.models.dits.nomas import NomasConfig
from sglang.multimodal_gen.configs.models.vaes.nomas import NomasVAEConfig
from sglang.multimodal_gen.runtime.models.dits.nomas import NomasTransformer3DModel
from sglang.multimodal_gen.runtime.models.vaes.nomas import NomasPixelUMMDecoder
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_model_parallel,destroy_distributed_environment
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args

root=Path('/task');out=root/'evidence/preflight/nomas-checkpoint-key-audit02.json';assert not out.exists()
assert not torch.cuda.is_initialized()
os.environ['LOCAL_RANK']='0';set_global_server_args(_make_unit_server_args())
init_distributed_environment(backend='gloo',distributed_init_method='file:///task/evidence/preflight/nomas-meta-audit02-rendezvous',timeout=60)
initialize_model_parallel(tensor_parallel_degree=1,sequence_parallel_degree=1)
report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),benchmark=False,scope='Actual frozen native models on meta; checkpoint header names/shapes only. No runtime/quality claim.',components={},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def alias(name):
 if name.startswith(('double_blocks.','txt_in.individual_token_refiner.blocks.')):
  return name.replace('.attn.','.')
 return name
try:
 for kind,config_cls,model_cls,folder in [('transformer',NomasConfig,NomasTransformer3DModel,'transformer'),('decoder',NomasVAEConfig,NomasPixelUMMDecoder,'vae')]:
  config=config_cls();config.update_model_arch(json.loads((root/'cache/nomas-native-common01'/folder/'config.json').read_text()))
  with torch.device('meta'):model=model_cls(config)
  expected=model.state_dict();base=root/'cache/nomas-frozen01/Hy-Nomas-Diffusers'/kind
  idx=json.loads(next(base.glob('*.index.json')).read_text())['weight_map'];mapper=get_param_names_mapping(model.param_names_mapping if kind=='transformer' else {},valid_target_names=set(expected))
  rows=[];mapped={};old_missing=[]
  with contextlib.ExitStack() as stack:
   readers={name:stack.enter_context(safe_open(str(base/name),framework='pt',device='cpu')) for name in set(idx.values())}
   for source,shard in idx.items():
    adapted=alias(source) if kind=='transformer' else source
    target,merge,split=mapper(adapted);assert merge is None and split is None
    assert target not in mapped,('duplicate',target)
    header=readers[shard].get_slice(source);shape=header.get_shape();dtype=header.get_dtype()
    assert target in expected,('unexpected_target',source,target)
    assert list(expected[target].shape)==shape,(source,shape,list(expected[target].shape))
    assert dtype=='BF16',source
    if mapper(source)[0] not in expected:old_missing.append(source)
    mapped[target]=source;rows.append(dict(source=source,adapted=adapted,target=target,shard=shard,shape=shape,dtype=dtype))
  assert set(mapped)==set(expected),sorted(set(expected)-set(mapped))
  report['components'][kind]=dict(tensors=len(rows),unique_complete_shape_match=True,original_missing_count=len(old_missing),original_missing=old_missing,mapping=rows)
 assert not torch.cuda.is_initialized();report['cuda_initialized']=False
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{q:v[q] for q in ['tensors','unique_complete_shape_match','original_missing_count']} for k,v in report['components'].items()}))
finally:
 destroy_model_parallel();destroy_distributed_environment()
