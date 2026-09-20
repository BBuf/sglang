"""Validate the candidate before its first unprofiled native saved request."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
while not (root/'validate-lingbot-full-norm.exit').exists():time.sleep(5)
assert (root/'validate-lingbot-full-norm.exit').read_text().strip()=='0'
repo = root / 'candidate-lingbot-current'
env = os.environ | {'CUDA_VISIBLE_DEVICES':'0', 'OMP_NUM_THREADS':'8',
    'PYTHONPATH':str(repo/'python'), 'FLASHINFER_DISABLE_VERSION_CHECK':'1'}
with (root/'gpu.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    selected = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    assert not any(line.startswith(selected) for line in processes.splitlines()), processes
    for label, command in [
        ('kernel-tests', ['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_lingbot_modulation.py']),
        ('scale-only-compat-tests', ['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_layernorm_modulate.py']),
        ('legacy-norm-tests', ['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_model_fast_paths.py','-k','ln_modulate or norm_modulate']),
        ('fp8-norm-tests', ['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_flux2_fp8_norm_quant.py']),
        ('runtime-tests', ['python','-m','pytest','-q','python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py','python/sglang/multimodal_gen/test/unit/test_diffusion_import_isolation.py']),
        ('marker', ['python','test/registered/kernels/benchmark/diffusion/bench_lingbot_modulation.py']),
    ]:
        with (root/f'lingbot-current-{label}.log').open('x') as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        (root/f'lingbot-current-{label}.exit').write_text(str(result.returncode))
        assert result.returncode == 0, label
with (root/'lingbot-current-admission.log').open('x') as log:
    result = subprocess.run(['python','-u',str(root/'run-lingbot-world-v2.py'),'--repo',str(repo),'--label','current-candidate-a1'],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'lingbot-current-admission.exit').write_text(str(result.returncode))
assert result.returncode == 0
base = json.loads((root/'artifacts/lingbot-world-v2/baseline-a1/result.json').read_text())
cand = json.loads((root/'artifacts/lingbot-world-v2/current-candidate-a1/result.json').read_text())
assert cand['output_sha256'] == base['output_sha256']
print(json.dumps({'baseline':base,'candidate':cand},indent=2))
