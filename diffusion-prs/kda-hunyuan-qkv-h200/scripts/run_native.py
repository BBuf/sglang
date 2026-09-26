#!/usr/bin/env python3
"""One fresh native SGLang request, pinned source/cache/GPU, complete artifacts."""
import argparse
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--model', choices=('zimage', 'wan21', 'sana-video', 'ltx2', 'flux2-klein', 'hunyuanvideo', 'ltx23'), required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--gpu', default='7')
    p.add_argument('--quality', choices=('lossless', 'high'), default='lossless')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--capture', action='store_true')
    p.add_argument('--shadow-baseline', help='Diagnostic-only pinned LTX2 ada9 reference repository; requires capture.')
    p.add_argument('--deterministic-audio-decode', action='store_true', help='Symmetric validation-only LTX audio cuDNN/TF32 reproduction context.')
    p.add_argument('--steps', type=int)
    p.add_argument('--frames', type=int)
    a = p.parse_args()
    assert not a.shadow_baseline or a.capture, '--shadow-baseline requires --capture'
    assert not a.deterministic_audio_decode or a.model == 'ltx23'
    root = Path(__file__).resolve().parent
    output_root = Path(a.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lease_root = Path(os.environ.get('KDA_GPU_LOCK_DIR', '/tmp/kda-gpu-locks'))
    lease_root.mkdir(parents=True, exist_ok=True)
    lock = (lease_root / f'gpu-{a.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    out = output_root / a.label
    out.mkdir(exist_ok=False)
    repo = Path(a.repo).resolve()
    config = json.loads((root / 'presets.json').read_text())[a.model]
    if a.steps is not None:
        config['steps'] = a.steps
    if a.frames is not None:
        config['frames'] = a.frames
    assert (repo / 'python/sglang/__init__.py').is_file()
    sys.path.insert(0, str(repo / 'python'))
    import sglang
    assert Path(sglang.__file__).resolve().is_relative_to(repo)
    env_script = repo / 'python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py'
    spec = importlib.util.spec_from_file_location('validation_env', env_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.get_repo_root() == repo
    module.check_write_access(repo)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    assigned = {line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() == a.gpu}
    assert len(assigned) == 1, inventory
    for _ in range(30):
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        if not any(line.split(',')[0].strip() in assigned for line in processes.splitlines()):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Assigned GPU has a foreign process; no process was killed')
    (out / 'gpu-before.txt').write_text(inventory + '\nProcesses:\n' + processes)
    source_files = sorted(set((repo / 'python/sglang/kernels/ops/diffusion').rglob('*.py')) | set((repo / 'python/sglang/multimodal_gen/runtime/models').rglob('*.py')))
    (out / 'source-files.json').write_text(json.dumps({str(path.relative_to(repo)): sha256(path) for path in source_files}, indent=2))
    versions = {}
    for distribution in ['torch', 'triton', 'diffusers', 'transformers', 'flash-attn', 'sglang', 'sgl-kernel']:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    versions['python'] = sys.version
    versions['installed_sglang_note'] = 'Distribution metadata may refer to installed package; imported source is explicitly pinned in request.json.'
    (out / 'environment.json').write_text(json.dumps(versions, indent=2))
    snapshot = Path('/cluster-storage/models') / ('models--' + config['model'].replace('/', '--')) / 'snapshots' / config['revision']
    snapshot = Path(config.get('local_model_path', snapshot))
    assert (snapshot / 'model_index.json').is_file(), snapshot
    env = os.environ.copy()
    env.update(PYTHONPATH=str(repo / 'python'), CUDA_VISIBLE_DEVICES=a.gpu,
               HF_HUB_CACHE='/cluster-storage/models', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               FLASHINFER_DISABLE_VERSION_CHECK='1', SGLANG_DIFFUSION_SYNC_STAGE_PROFILING='1', OMP_NUM_THREADS='8')
    if a.capture:
        env['PYTHONPATH'] = str(root / 'capture') + ':' + env['PYTHONPATH']
        env['KDA_CAPTURE_DIR'] = str(out / 'calls')
        if a.shadow_baseline:
            env['KDA_SHADOW_BASELINE'] = str(Path(a.shadow_baseline).resolve())
    if a.deterministic_audio_decode:
        env['PYTHONPATH'] = str(root / 'repro') + ':' + str(repo / 'python')
        env['KDA_DETERMINISTIC_AUDIO_DECODE'] = '1'
        env['KDA_REPRO_DIR'] = str(out / 'repro')
    if a.profile:
        env['SGLANG_DIFFUSION_TORCH_PROFILER_DIR'] = str(out / 'traces')
    cmd = ['sglang', 'generate', '--backend=sglang', '--model-path=' + str(snapshot), '--revision=' + config['revision'],
           '--prompt=' + config['prompt'], '--seed=' + str(config['seed']), '--width=' + str(config['width']), '--height=' + str(config['height']),
           '--num-frames=' + str(config['frames']), '--num-inference-steps=' + str(config['steps']), '--guidance-scale=' + str(config['guidance']),
           '--num-gpus=1', '--tp-size=1', '--ulysses-degree=1', '--master-port=' + str(32000 + int(a.gpu) * 10), '--scheduler-port=' + str(32100 + int(a.gpu) * 10), '--port=' + str(32200 + int(a.gpu) * 10), '--performance-mode=manual', '--enable-torch-compile=false',
           '--dit-cpu-offload=false', '--dit-layerwise-offload=false', '--vae-cpu-offload=false', '--text-encoder-cpu-offload=false',
           '--quality=' + a.quality, '--warmup-mode=request', '--warmup-resolutions=' + str(config['width']) + 'x' + str(config['height']),
           '--warmup-num-frames=' + str(config['frames']), '--save-output', '--output-path=' + str(out), '--output-file-name=sample',
           '--perf-dump-path=' + str(out / 'perf.json')]
    cmd.extend(config.get('extra_args', []))
    if config.get('fps'):
        cmd.append('--fps=' + str(config['fps']))
    if a.profile:
        cmd.extend(['--profile', '--profile-all-stages'])
    metadata = dict(source_sha=a.source_sha, repo=str(repo), sglang_file=sglang.__file__, request=config, quality=a.quality,
                    gpu=a.gpu, command=cmd, diagnostic=bool(a.profile or a.capture), captured=a.capture, profiled=a.profile, shadow_baseline=a.shadow_baseline,
                    deterministic_audio_decode=a.deterministic_audio_decode,
                    validation_repro_source_sha256=sha256(root / 'repro/sitecustomize.py') if a.deterministic_audio_decode else None,
                    environment={key: env[key] for key in ['PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'FLASHINFER_DISABLE_VERSION_CHECK', 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING']})
    (out / 'request.json').write_text(json.dumps(metadata, indent=2))
    start = time.time()
    forbidden = []
    with (out / 'native.log').open('w') as log:
        proc = subprocess.Popen(cmd, env=env, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        (out / 'child.pid').write_text(str(proc.pid))
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
            if any(marker in line for marker in ['Falling back to diffusers backend', 'Using diffusers backend', 'Loaded diffusers pipeline']):
                forbidden.append(line.strip())
                proc.terminate()
        code = proc.wait()
    artifacts = sorted(path for path in out.glob('sample*') if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.mp4', '.webp'})
    perf_path = out / 'perf.json'
    result = dict(**metadata, returncode=code, elapsed_with_load_s=time.time() - start, native_fallback_signals=forbidden,
                  artifacts=[dict(path=str(path), sha256=sha256(path), size=path.stat().st_size) for path in artifacts])
    result['valid'] = code == 0 and not forbidden and bool(artifacts) and perf_path.is_file()
    if a.deterministic_audio_decode:
        records = [json.loads(line) for path in (out / 'repro').glob('audio-context-*.jsonl') for line in path.read_text().splitlines()]
        result['audio_repro_context_calls'] = records
        result['valid'] &= all(any(row['method'].endswith(method) for row in records) for method in ['AutoencoderKLLTX2Audio.decode', 'LTX2Vocoder.forward'])
    if perf_path.is_file():
        perf = json.loads(perf_path.read_text())
        result['worker_e2e_s'] = perf.get('total_duration_ms', 0) / 1000
        result['denoise_s'] = sum(x.get('duration_ms', 0) for x in perf.get('steps', []) if x.get('name', '').endswith(('DenoisingStage', 'RefinementStage')) and 'BeforeDenoisingStage' not in x.get('name', '')) / 1000
        result['decode_s'] = sum(x.get('duration_ms', 0) for x in perf.get('steps', []) if x.get('name', '').endswith('DecodingStage')) / 1000
    log_text = (out / 'native.log').read_text()
    clean_log = re.sub(r'\x1b\[[0-9;]*m', '', log_text)
    saved_timers = re.findall(r'Pixel data generated successfully in ([0-9.]+) seconds', clean_log)
    result['client_saved_e2e_s'] = float(saved_timers[-1]) if saved_timers else None
    result['client_saved_timer_precision_s'] = 0.01
    if not a.profile and not a.capture:
        result['valid'] &= bool(saved_timers)
    if a.profile:
        result['traces'] = [str(path) for path in (out / 'traces').rglob('*.trace.json*')]
        result['valid'] &= bool(result['traces'])
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    return 0 if result['valid'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
