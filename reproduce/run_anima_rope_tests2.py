"""Existing regression tests on frozen main and single-file candidate, GPU0."""
import hashlib,json,subprocess,time
from pathlib import Path
R=Path('/data/xyzhang/diffusion-rtx6000-20261026')
D=R/'block-runs/anima-rope-tests2'
D.mkdir(exist_ok=False)
rows=[]
for arm,repo,expected in [('candidate','/task/candidates/anima-rope-rounding','f6c554a8d281cd2847925f5c5e01cd82995c73c57ef5b1ed78cee356620ea875')]:
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    base=['docker','exec','-e','CUDA_VISIBLE_DEVICES=0','-e','FLASHINFER_DISABLE_VERSION_CHECK=1','-e','PYTHONPATH='+repo+'/python','-w',repo,'bbuf-diffusion-rtx6000-20261026-sgl']
    patch=subprocess.check_output(base+['git','diff','HEAD','--binary','--full-index'])
    assert hashlib.sha256(patch).hexdigest()==expected
    with (D/(arm+'-preflight.log')).open('xb') as f:
        subprocess.run(['docker','exec','-e','PYTHONPATH=/task/repos/sglang/python','-w','/task/repos/sglang','bbuf-diffusion-rtx6000-20261026-sgl','python3','/task/preflight_ming_environment.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
    cmd=base+['python3','-m','pytest','-v','test/registered/kernels/ops/diffusion/test_rope.py','test/registered/kernels/ops/diffusion/test_model_fast_paths.py','-k','anima']
    start=time.monotonic()
    with (D/(arm+'.log')).open('xb') as f:
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    rows.append(dict(arm=arm,command=cmd,patch_sha256=expected,returncode=result.returncode,elapsed_seconds=time.monotonic()-start))
    (D/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows[-1]),flush=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
