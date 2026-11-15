"""Adapt only safetensors header keys; prove payload bytes unchanged."""
import datetime
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

root=Path('/task');old=root/'cache/nomas-native-common01';out=root/'cache/nomas-native-common02'
assert not out.exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
audit=json.loads((root/'evidence/preflight/nomas-checkpoint-key-audit02.json').read_text())
assert all(x['unique_complete_shape_match'] for x in audit['components'].values())
assert audit['components']['transformer']['tensors']==2268 and audit['components']['decoder']['tensors']==476
mapping={x['source']:x['adapted'] for x in audit['components']['transformer']['mapping']}
assert len(mapping)==len(set(mapping.values()))==2268
shutil.copytree(old,out,symlinks=True)
base=root/'cache/nomas-frozen01/Hy-Nomas-Diffusers/transformer';files=[]
for source in sorted(base.glob('*.safetensors')):
 target=out/'transformer'/source.name;assert target.is_symlink();target.unlink()
 partial=target.with_suffix('.partial');hasher=hashlib.sha256()
 with source.open('rb') as src,partial.open('xb') as dst:
  old_length=struct.unpack('<Q',src.read(8))[0];raw=src.read(old_length);header=json.loads(raw)
  converted={mapping[k] if k!='__metadata__' else k:v for k,v in header.items()}
  assert len(converted)==len(header)
  encoded=json.dumps(converted,separators=(',',':'),ensure_ascii=False).encode();encoded+=b' '*((-len(encoded))%8)
  dst.write(struct.pack('<Q',len(encoded)));dst.write(encoded)
  for block in iter(lambda:src.read(8*1024**2),b''):hasher.update(block);dst.write(block)
 data_sha=hashlib.sha256()
 with partial.open('rb') as dst:
  size=struct.unpack('<Q',dst.read(8))[0];new_header=json.loads(dst.read(size));assert new_header==converted
  for block in iter(lambda:dst.read(8*1024**2),b''):data_sha.update(block)
 assert hasher.hexdigest()==data_sha.hexdigest()
 partial.rename(target)
 files.append(dict(source=str(source),target=str(target),tensors=len(header)-('__metadata__' in header),source_header_sha256=hashlib.sha256(raw).hexdigest(),target_header_sha256=hashlib.sha256(encoded).hexdigest(),source_data_offset=8+old_length,target_data_offset=8+len(encoded),payload_bytes=source.stat().st_size-8-old_length,payload_sha256=hasher.hexdigest(),target_payload_sha256=data_sha.hexdigest(),numeric_payload_byte_exact=True))
 print(json.dumps(files[-1]),flush=True)
index=next(base.glob('*.index.json'));j=json.loads(index.read_text());j['weight_map']={mapping[k]:v for k,v in j['weight_map'].items()};target=out/'transformer'/index.name;assert target.is_symlink();target.unlink();target.write_text(json.dumps(j,indent=2)+'\n')
receipt=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),benchmark=False,shared=False,canonical_revision='a1ae04d11ae4ea176d1f76de634c115944f9882d',base_weight_identity='evidence/preflight/nomas-common-weight-identity01.json',key_shape_audit='evidence/preflight/nomas-checkpoint-key-audit02.json',files=files,changed_tensor_names=sum(k!=v for k,v in mapping.items()),mapping=mapping,source_modified=False,runtime_source_modified=False,numeric_dtype_or_layout_changed=False,runtime_validated=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'task-ownership.json').write_text(json.dumps(receipt,indent=2)+'\n');(root/'evidence/preflight/nomas-key-only-materialization02.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(files=len(files),renamed=receipt['changed_tensor_names'],payload_byte_exact=True)),flush=True)
