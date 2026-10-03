import hashlib,json,pathlib,subprocess
r=pathlib.Path('/scratch/cold-l2')
m=json.loads((r/'manifest-final.json').read_text())
bad=[p for p,h in m['files'].items() if hashlib.sha256((r/'candidate-final'/p).read_bytes()).hexdigest()!=h]
assert not bad,bad
result=dict(candidate=m['candidate'],baseline=m['baseline'],verified_files=len(m['files']),mismatches=bad)
base=json.loads((r/'baseline-manifest.json').read_text())
bad_base=[]
for name, expected in base.items():
 data=(r/'baseline'/name).read_bytes()
 actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if actual!=expected:bad_base.append(name)
assert not bad_base,bad_base
result.update(baseline_verified_files=len(base),baseline_mismatches=bad_base)
(r/'final-results/source-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
