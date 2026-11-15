"""Prove common tensors, then expose pinned BF16 weights to native SGLang."""
import contextlib
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

root = Path('/task')
assert not torch.cuda.is_initialized()
assert not subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True
).strip(), 'Materialize only while the task GPUs are idle.'
pilot = json.loads((root / 'evidence/preflight/h3-image-row-overlap-pilot02-status.json').read_text())
assert pilot['status'] == 'profile_and_quiet_pilot_complete_review_pending'
assert not pilot['benchmark']
plan = json.loads((root / 'inputs/nomas-weight-staging01.json').read_text())
cache = root / 'cache/nomas-frozen01'
out = root / 'cache/nomas-native-common01'
assert not out.exists()
verified = []
for item in plan['files']:
    path = cache / item['directory'] / item['path']
    assert path.stat().st_size == item['size'], path
    digest = hashlib.sha256()
    hf = hashlib.sha256() if item['hash_type'] == 'sha256' else hashlib.sha1(
        b'blob ' + str(item['size']).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
            hf.update(block)
    assert hf.hexdigest() == item['expected'], path
    verified.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=item['size']))

canonical = cache / 'Hy-Nomas-Diffusers'
original = cache / 'Qwen2.5-VL-7B-Instruct'
converted_index = json.loads((canonical / 'text_encoder/model.safetensors.index.json').read_text())['weight_map']
original_index = json.loads((original / 'model.safetensors.index.json').read_text())['weight_map']
assert len(converted_index) == 338
assert set('model.' + name for name in converted_index) == {
    name for name in original_index if name.startswith('model.')
}
tensor_receipts = []
with contextlib.ExitStack() as stack:
    paths = {(canonical / 'text_encoder' / name) for name in converted_index.values()}
    paths.update(original / original_index['model.' + name] for name in converted_index)
    readers = {path: stack.enter_context(safe_open(str(path), framework='pt', device='cpu')) for path in paths}
    for name, shard in converted_index.items():
        left = readers[canonical / 'text_encoder' / shard].get_tensor(name)
        right = readers[original / original_index['model.' + name]].get_tensor('model.' + name)
        assert left.shape == right.shape and left.dtype == right.dtype == torch.bfloat16, name
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8)), name
        tensor_receipts.append(dict(name=name, shape=list(left.shape), dtype=str(left.dtype),
                                    sha256=hashlib.sha256(left.contiguous().view(torch.uint8).numpy()).hexdigest()))

# No casting or tensor rewriting: both native frameworks read the canonical
# DiT/decoder shards; SGLang's existing full-Qwen loader needs the original wrapper.
overlay = root / 'inputs/nomas-metadata01/Hy-Nomas-overlay'
out.mkdir()
(out / 'model_index.json').write_bytes((overlay / 'model_index.json').read_bytes())
for destination, source in [('transformer', canonical / 'transformer'),
                            ('vae', canonical / 'decoder'),
                            ('text_encoder', original), ('tokenizer', canonical / 'tokenizer')]:
    folder = out / destination
    folder.mkdir()
    for path in source.iterdir():
        if path.is_file() and (path.name.endswith(('.safetensors', '.json', '.jinja', '.txt'))):
            if path.name == 'config.json' and destination in ('transformer', 'vae', 'text_encoder'):
                continue
            (folder / path.name).symlink_to(path)
for name in ('transformer', 'vae'):
    (out / name / 'config.json').write_bytes((overlay / name / 'config.json').read_bytes())
text_config = json.loads((original / 'config.json').read_text())
text_config['text_len'] = 1280
(out / 'text_encoder/config.json').write_text(json.dumps(text_config, indent=2) + '\n')
(out / 'scheduler').mkdir()
(out / 'scheduler/scheduler_config.json').write_bytes((canonical / 'scheduler/scheduler_config.json').read_bytes())
receipt = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), task='diffusion-6000-20261113',
               shared=False, model_revision=plan['canonical_model_revision'],
               original_text_encoder_revision=plan['original_text_encoder_revision'],
               files_freshly_verified=verified, text_encoder_tensor_identity=tensor_receipts,
               transformer_decoder='Unmodified symlinks to canonical BF16 shard bytes',
               native_SGLang_layout=str(out), native_Omni_layout=str(canonical),
               runtime_validation=False, generated_quality_validation=False,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out / 'task-ownership.json').write_text(json.dumps(receipt, indent=2) + '\n')
(root / 'evidence/preflight/nomas-common-weight-identity01.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(files_verified=len(verified), text_tensors_exact=len(tensor_receipts),
                      runtime_validated=False, SGLang_layout=str(out))))
