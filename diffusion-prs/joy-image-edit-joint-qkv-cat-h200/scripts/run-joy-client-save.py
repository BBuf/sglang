"""Native DiffGenerator: one full saved warmup, then five saved measured requests."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    root = Path('/campaign')
    lock = (root / 'gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    repo = Path(args.repo).resolve()
    os.environ.update(json.loads((root / 'artifacts/joy-image-edit-cache-env.json').read_text()))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=str(repo / 'python'), OMP_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='0,1',
        SGLANG_DIFFUSION_SYNC_STAGE_PROFILING='1')
    sys.path.insert(0, str(repo / 'python'))
    os.chdir(repo)
    import sglang
    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
    from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
        add_multimodal_gen_generate_args, _resolve_cli_sampling_params_cls, maybe_dump_performance)
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser
    assert Path(sglang.__file__).resolve().is_relative_to(repo)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    selected = {line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0', '1'}}
    for _ in range(35):
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Assigned GPUs busy; no process killed')
    for argv in [['git', 'diff', '--quiet'], ['git', 'diff', '--cached', '--quiet']]:
        subprocess.run(argv, check=True)
    out = root / 'artifacts/joy-image-edit/client-save' / args.label
    out.mkdir(parents=True, exist_ok=False)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (out / 'source.txt').write_text(sha + '\n' + str(sglang.__file__) + '\n')
    (out / 'gpu-before.txt').write_text(inventory + '\nProcesses:\n' + processes)
    preset = json.loads((root / 'artifacts/joy-image-edit/outplace-fullwarm-r1-baseline-a1/preset.json').read_text())
    argv = ['--backend=sglang', '--model-path=' + preset['path'], '--prompt=' + preset['prompt'],
        '--seed=42', '--image-path=' + preset['image_path'], *preset['extra_args'],
        '--quality=lossless', '--save-output', '--warmup-mode=request']
    cli = add_multimodal_gen_generate_args(FlexibleArgumentParser())
    cli_args, unknown = cli.parse_known_args(argv)
    assert not unknown, unknown
    # Native CLI uses sys.argv to identify explicitly supplied flags.
    sys.argv = [sys.argv[0], *argv]
    server_args = ServerArgs.from_cli_args(cli_args, unknown)
    sampling = _resolve_cli_sampling_params_cls(server_args).get_cli_args(cli_args)
    (out / 'protocol.json').write_text(json.dumps(dict(argv=argv, saved_warmups=1, measured_requests=5,
        native_request_warmup_steps=40, source=sha, return_file_paths_only=False, output_transport='native client save including returned pixel data', client_wall_definition='perf_counter around generate plus CLI maybe_dump_performance; hashing and campaign JSON excluded'), indent=2))
    generator = DiffGenerator.from_pretrained(model_path=server_args.model_path, server_args=server_args, local_mode=True)
    records = []
    try:
        for index in range(6):
            role = 'saved-warmup' if index == 0 else 'measured'
            perf = out / f'request-{index}-perf.json'
            params = dict(sampling, request_id=generate_request_id(), output_path=str(out),
                output_file_name=f'request-{index}.png', perf_dump_path=str(perf), return_file_paths_only=False)
            cli_args.perf_dump_path = str(perf)
            print(f'CAMPAIGN_REQUEST_START {index} {role}', flush=True)
            started = time.perf_counter()
            result = generator.generate(sampling_params_kwargs=params)
            maybe_dump_performance(cli_args, server_args, preset['prompt'], result)
            wall = time.perf_counter() - started
            if isinstance(result, list):
                assert len(result) == 1
                result = result[0]
            assert result is not None and Path(result.output_file_path).exists()
            metrics = json.loads(perf.read_text())
            digest = hashlib.sha256(Path(result.output_file_path).read_bytes()).hexdigest()
            assert digest == '76cfd40eb83feacb1b0292e171b9beb2a8cd7eb7fadc606cb1632e886d960ce6'
            row = dict(index=index, role=role, source=sha, worker_s=metrics['total_duration_ms']/1000,
                denoise_s=next(s['duration_ms']/1000 for s in metrics['steps'] if s['name'] == 'DenoisingStage'),
                client_wall_s=wall, output_sha256=digest, output_file_path=result.output_file_path,
                peak_gib=max(s.get('peak_reserved_mb', 0) for s in metrics['memory_checkpoints'].values())/1024)
            records.append(row)
            (out / 'records.json').write_text(json.dumps(records, indent=2))
            print('CAMPAIGN_REQUEST_END ' + json.dumps(row), flush=True)
    finally:
        generator.shutdown()


if __name__ == '__main__':
    main()
