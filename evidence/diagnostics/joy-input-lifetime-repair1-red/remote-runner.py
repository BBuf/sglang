
import datetime, hashlib, json, os, pathlib, subprocess, sys, time, xml.etree.ElementTree as ET
request=json.load(sys.stdin)
root=pathlib.Path('/task')
phase=request['phase']; assert phase in ['red','green']
run='joy-input-lifetime-repair1-'+phase
folder=root/'outputs/diagnostics'/run
folder.mkdir(exist_ok=False)
repo=root/'candidates'/run
base='b08c24e9630c63790c9f86b48d6c7fc10a0c4301'
def cmd(args): return subprocess.check_output(args,text=True).strip()
assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
assert cmd(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'])=='GPU-1565b45e-a702-2015-9776-61f68c36fd59'
subprocess.run(['git','-C',str(root/'repos/sglang'),'worktree','add','--detach',str(repo),base],check=True,stdout=sys.stderr)
patch=request['patch'].encode()
assert hashlib.sha256(patch).hexdigest()==request['patch_sha256']
(folder/'source.patch').write_bytes(patch)
subprocess.run(['git','-C',str(repo),'apply','--check','-'],input=patch,check=True)
subprocess.run(['git','-C',str(repo),'apply','-'],input=patch,check=True)
assert subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--'])==patch
env=os.environ.copy()
env.update(PYTHONPATH=str(repo/'python'),FLASHINFER_DISABLE_VERSION_CHECK='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='0')
with (folder/'preflight.log').open('xb') as stream:
    subprocess.run([sys.executable,str(root/'preflight_diffusion_current_head1.py'),str(repo)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
test='test/registered/kernels/ops/diffusion/test_model_fast_paths.py::test_joy_image_mlp_quality_restore_and_dynamic_lora'
junit=folder/'junit.xml'
args=[sys.executable,'-m','pytest','-q',test,'--junitxml='+str(junit)]
start=time.monotonic()
with (folder/'pytest.log').open('xb') as stream:
    completed=subprocess.run(args,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
elapsed=time.monotonic()-start
cases=list(ET.parse(junit).getroot().iter('testcase'))
assert len(cases)==1 and cases[0].attrib['name']==test.split('::')[-1]
assert cases[0].find('skipped') is None and cases[0].find('error') is None
failure=cases[0].find('failure')
if phase=='red':
    assert completed.returncode==1 and failure is not None
    assert 'Joy MLP retained its temporary input' in ''.join(failure.itertext())
else:
    assert completed.returncode==0 and failure is None
for _ in range(30):
    if not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']): break
    time.sleep(1)
assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
receipt=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,phase=phase,base=base,
    patch_sha256=request['patch_sha256'],seconds=elapsed,command=args,returncode=completed.returncode,
    tests=1,skipped=0,expected_result_observed=True,gpu_idle_after=True,
    scope='Actual temporary-input release and numerical/quality/compile/dynamic-LoRA regression. Not model performance admission.')
files={p.name:p.read_text() for p in folder.iterdir() if p.is_file()}
receipt['files_sha256']={k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}
(folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(receipt=receipt,files=files)))
