"""Download a task-owned public mirror checkpoint, pinned by original HF hashes."""
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path('/task')
plan=json.loads(Path(sys.argv[1]).read_text())
dest=Path(plan['destination'])
assert dest.parent==ROOT/'cache/common' and not dest.is_symlink()
evidence=ROOT/'evidence/downloads'
manifest=evidence/(plan['label']+'-modelscope-download-manifest.json')
assert not manifest.exists(), 'Completed downloads must not be overwritten'
chunk_size=1024**3
records=[]


def digest(path,row):
    sha=hashlib.sha256();git=hashlib.sha1(f'blob {row["bytes"]}\0'.encode());count=0
    with path.open('rb') as stream:
        while block:=stream.read(8*1024**2):sha.update(block);git.update(block);count+=len(block)
    assert count==row['bytes'],(row['path'],count)
    assert (sha.hexdigest()==row['hf_lfs_sha256'] if row['hf_lfs_sha256'] else git.hexdigest()==row['hf_blob']),row['path']
    return sha.hexdigest()


def endpoint(row):
    query=urllib.parse.urlencode(dict(Revision=row['modelscope_revision'],FilePath=row['path']))
    return 'https://modelscope.cn/api/v1/models/'+plan['repo']+'/repo?'+query


def fetch(job):
    row,start,end,path=job
    path.parent.mkdir(parents=True,exist_ok=True)
    # Parts are new task-owned files. Final assembly must still pass the full
    # original HF digest, including any part reused after interruption.
    if path.exists() and path.stat().st_size==end-start+1:return dict(path=str(path),reused=True)
    partial=path.with_name(path.name+'.partial')
    for attempt in range(3):
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req=urllib.request.Request(endpoint(row),headers={'Range':f'bytes={start}-{end}'})
            with opener.open(req,timeout=120) as response,partial.open('wb') as out:
                if response.status==206:
                    assert response.headers['Content-Range']==f'bytes {start}-{end}/{row["bytes"]}'
                else:assert response.status==200 and start==0 and end==row['bytes']-1
                count=0
                while block:=response.read(8*1024**2):
                    count+=len(block);assert count<=end-start+1
                    out.write(block)
            assert count==end-start+1
            partial.replace(path)
            print(json.dumps(dict(event='range_complete',file=row['path'],start=start,end=end,bytes=count)),flush=True)
            return dict(path=str(path),reused=False)
        except Exception as exc:
            print(json.dumps(dict(event='range_failed',file=row['path'],start=start,attempt=attempt+1,error=repr(exc))),flush=True)
            if attempt==2:raise
            time.sleep(2**attempt)


def parts_for(row):
    target=dest/row['path']
    assert target.resolve().is_relative_to(dest.resolve()) and not target.is_symlink()
    if target.exists():
        sha=digest(target,row)
        records.append(dict(path=row['path'],bytes=row['bytes'],sha256=sha,verified_against_hf=True,reused_existing_rehashed_file=True))
        return []
    folder=dest/'.download-parts'/hashlib.sha256(row['path'].encode()).hexdigest()[:24]
    return [(row,start,min(start+chunk_size,row['bytes'])-1,folder/f'{index:05d}.part') for index,start in enumerate(range(0,row['bytes'],chunk_size))]


def assemble(row,jobs):
    if not jobs:return
    target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_name(target.name+'.partial')
    with partial.open('wb') as out:
        for _,_,_,path in jobs:
            with path.open('rb') as stream:shutil.copyfileobj(stream,out,8*1024**2)
    sha=digest(partial,row);partial.replace(target)
    for _,_,_,path in jobs:path.unlink()
    jobs[0][3].parent.rmdir()
    records.append(dict(path=row['path'],bytes=row['bytes'],sha256=sha,verified_against_hf=True,reused_existing_rehashed_file=False))
    print(json.dumps(dict(event='file_verified',file=row['path'],bytes=row['bytes'],sha256=sha)),flush=True)


started=time.time()
for is_weight in [False,True]:
    group=[row for row in plan['files'] if (row['bytes']>=20*1024**2)==is_weight]
    jobs_by_file=[(row,parts_for(row)) for row in group]
    jobs=[job for _,jobs in jobs_by_file for job in jobs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(fetch,jobs):pass
    for row,jobs in jobs_by_file:assemble(row,jobs)
    if not is_weight:
        (evidence/(plan['label']+'-metadata-download-manifest.json')).write_text(json.dumps(dict(files=records,weights_downloaded=False),indent=2)+'\n')
assert len(records)==len(plan['files'])
manifest.write_text(json.dumps(dict(hf_repo=plan['repo'],hf_revision=plan['revision'],destination=str(dest),
    complete=True,files=records,environment_proxy=False,credentials_sent=False,
    plan_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.time()-started),indent=2)+'\n')
parts=dest/'.download-parts'
if parts.exists():parts.rmdir()
print(json.dumps(dict(event='complete',files=len(records),bytes=sum(x['bytes'] for x in records))),flush=True)
