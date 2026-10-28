"""Download public ModelScope files directly, enforcing the frozen HF hashes."""

import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

TASK = Path('/task')
EVIDENCE = TASK / 'evidence/downloads'
HF = json.loads((EVIDENCE / 'qwen21-hf-pinned-blobs.json').read_text())
MS = {x['Path']: x for x in json.loads((EVIDENCE / 'qwen21-modelscope-files.json').read_text())['Data']['Files']}
DEST = TASK / 'cache/common/Qwen-Image-2.1'


def fetch(item):
    name = item['rfilename']
    mirror = MS[name]
    if 'lfs' in item:
        assert mirror['Sha256'] == item['lfs']['sha256'], name
    assert mirror['Size'] == item['size'], name
    dest = DEST / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == item['size']:
        start = time.monotonic()
        sha = hashlib.sha256()
        git = hashlib.sha1(f"blob {item['size']}\0".encode())
        with dest.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                sha.update(chunk)
                git.update(chunk)
        matches = (sha.hexdigest() == item['lfs']['sha256'] if 'lfs' in item
                   else git.hexdigest() == item['blobId'])
        if matches:
            row = {'path': name, 'bytes': item['size'], 'sha256': sha.hexdigest(),
                   'hf_revision': HF['sha'], 'modelscope_file_revision': mirror['Revision'],
                   'verified_against_hf': True, 'reused_existing_rehashed_file': True,
                   'seconds': time.monotonic() - start, 'environment_proxy': False}
            print(json.dumps(row), flush=True)
            return row
    params = urllib.parse.urlencode({'Revision': mirror['Revision'], 'FilePath': name})
    url = 'https://modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1/repo?' + params
    # No implicit proxy, login cookies, or HF token sent to the mirror.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.monotonic()
    tmp = dest.with_name(dest.name + '.partial')
    for attempt in range(3):
        try:
            sha = hashlib.sha256()
            git = hashlib.sha1(f"blob {item['size']}\0".encode())
            count = 0
            with opener.open(url, timeout=120) as response, tmp.open('wb') as out:
                final_host = urllib.parse.urlsplit(response.url).hostname
                while chunk := response.read(8 * 1024 * 1024):
                    out.write(chunk)
                    sha.update(chunk)
                    git.update(chunk)
                    count += len(chunk)
            assert count == item['size'], (name, count, item['size'])
            if 'lfs' in item:
                assert sha.hexdigest() == item['lfs']['sha256'], name
            else:
                assert git.hexdigest() == item['blobId'], (name, 'HF git blob differs')
            tmp.replace(dest)
            row = {'path': name, 'bytes': count, 'sha256': sha.hexdigest(),
                   'hf_revision': HF['sha'], 'modelscope_file_revision': mirror['Revision'],
                   'verified_against_hf': True, 'final_host': final_host,
                   'seconds': time.monotonic() - start, 'environment_proxy': False}
            print(json.dumps(row), flush=True)
            return row
        except Exception as exc:
            print(json.dumps({'path': name, 'attempt': attempt + 1, 'error': str(exc)}), flush=True)
            if attempt == 2:
                raise


if __name__ == '__main__':
    files = [x for x in HF['siblings'] if x['rfilename'].startswith(('processor/', 'text_encoder/', 'tokenizer/', 'transformer/', 'vae/', 'scheduler/')) or x['rfilename'] == 'model_index.json']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, files))
    (EVIDENCE / 'qwen21-modelscope-download-manifest.json').write_text(json.dumps({
        'hf_repo': HF['id'], 'hf_revision': HF['sha'],
        'modelscope_repo': 'Qwen/Qwen-Image-2.1', 'destination': str(DEST),
        'complete': True, 'files': records,
    }, indent=2) + '\n')
