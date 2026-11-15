"""Append verified scope evidence to the existing separate evidence checkout."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/'evidence/public/nomas-mean-37999'
assert (out/'.git').is_dir()
assert not subprocess.check_output(['git','-C',str(out),'status','--porcelain'])
assert subprocess.check_output(['git','-C',str(out),'rev-parse','HEAD'],text=True).strip()=='9473cba216404611937d488804a56682f83882e7'
summary=json.loads((root/'evidence/quality/nomas-241-main-cleanup25/summary.json').read_text())
assert len(summary['pairs'])==7 and all(x['artifact_exact'] and x['decoded_rgb_exact'] for x in summary['pairs'])
assert summary['visual_review']['status']=='approved_main_PR_representative_panels'
rows=[]


def copy(path):
    rel=path.relative_to(root)
    assert path.is_file() and not path.is_symlink() and path.stat().st_size<95*1024**2
    target=out/rel
    if target.exists():
        assert target.read_bytes()==path.read_bytes(),rel
        return
    if path.suffix in ['.json','.jsonl','.md','.log','.txt','.py','.yaml']:
        assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|https?://[^\s/:]+:[^\s/@]+@)',path.read_text(errors='replace')),rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,target)
    rows.append(dict(path=str(rel),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


for name in ['nomas-241-native17','nomas-241-main-cleanup25','nomas-sp2-native18']:
    for p in (root/'evidence/quality'/name).rglob('*'):
        if p.is_file() and p.suffix in ['.json','.png','.log']:copy(p)
for name in ['nomas-241-sgl-native17','nomas-241-omni-native17','nomas-241-sgl-mean-quality17',
             'nomas-sp2-sgl-native18','nomas-sp2-omni-native18','nomas-sp2-PR-native18',
             'nomas-c2-sgl-native23','nomas-c2-omni-native23','nomas-c2-PR-native23']:
    for p in (root/'runs'/name).rglob('*'):
        if p.is_file() and 'server-storage' not in p.parts and p.suffix in ['.json','.jsonl','.log','.txt']:
            copy(p)
for pair in summary['pairs']:
    for name in pair['paths']:copy(root/name)
for pattern in ['nomas-sp2-*.json','nomas-c2-*.json','nomas-241-*.json','nomas-sp2-deploy18.yaml']:
    for p in (root/'inputs').glob(pattern):copy(p)
for name in ['nomas-sp2-visits18','nomas-c2-visits23','nomas-sp2-observer18','nomas-sp2-candidate-observer18']:
    for p in (root/'scripts'/name).glob('*.py'):copy(p)
for name in ['run_nomas_sp2_18.py','run_nomas_c2_23.py','audit_nomas_sp2_18.py','prepare_nomas_mean_media25.py','prepare_nomas_supplement26.py']:
    copy(root/'scripts'/name)
for name in ['nomas-extended-summary26.json','nomas-sp2_18-controller.json','nomas-c2_23-controller.json','nomas-matrix17-controller.json']:
    copy(root/'evidence/preflight'/name)

readme='''# Additional scope checks for PR50602, unchanged head37999f7a

Frozen main0745bfe4, Omni aa2968e8, final PR37999f7a. Same physical RTX PRO6000 ServerEdition GPUs, native backend, canonical weights and fixed inputs. This supplement adds correctness/serving evidence; it adds **no performance or BCG claim**. The quiet49-frame ABBA and cleanup results in the original evidence commit remain the only model performance claim.

| Additional scope | Actual main/PR agreement | Native execution |
|---|---|---|
| Single GPU,241frames,short/long,3seeds+repeat |7MP4pairs and1687RGBframes exact;SSIM1,PSNR infinity,max error0 |4332fused mean calls/request |
| Ulysses2,49/121frames,short/long+repeat |6MP4pairs exact same-budget main; each framework also exact its own SP1 outputs |2actual rank receipts;361/1083fused calls per rank for49/121frames |
| Two outstanding API requests,49frames,3pairs |6PR outputs exact main C1/C2; both frameworks'6C2 outputs exact own C1 |Two nonterminal jobs observed after both POSTs;722fused calls/PR request |

The original17main/PR media pairs plus19additional pairs give36byte-identical pairs. No assumption of kernel concurrency or throughput benefit follows from two outstanding API jobs. Shared-code compatibility is proven only for these exercised combinations.

Cross-framework quality remains excluded from performance admission: all6new241-frame pairs fail the unchangedSSIM>=0.92/PSNR>=24dB gate, bringing the SP1 total to18failed pairs; all4SP2 cross pairs fail as well. Six actual241-frame worst panels were inspected: crop/geometry,cables/spokes,text and shadows differ. Both arms producing plausible bicycles does not waive the numeric failures. Native noise/masks/schedules match. No single-cause attribution or Omni speedup is claimed.

Actual first/middle/last main/PR panels are in `evidence/quality/nomas-241-main-cleanup25/`; full decoded comparisons cover all241frames. Their `summary.json` maps to real generated MP4s under the two241-frame run directories. SP2/C2 output hashes link to identical native SP1 content in the original evidence archive; per-request raw replies,rank observations and concurrent status snapshots are included. Full private runtime tensors and media are retained on the task machine and archived separately; no weights are included here.
'''
(out/'additional-scope26.md').write_text(readme)
(out/'additional-scope26-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),path=str(out))))
