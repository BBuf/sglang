"""Native serving measurements with explicit diagnostic tagging and ownership evidence."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import pynvml as nv
import requests


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['sgl', 'omni'])
    parser.add_argument('--contract', type=Path, default=Path('/task/inputs/anima-t2i-workloads.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allowed-gpu-pids', type=int, nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--configuration-label', default='eager')
    parser.add_argument('--prompt-id', choices=['short', 'long', 'chinese'], required=True)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--expected-sha')
    parser.add_argument('--expected-diff-sha256')
    parser.add_argument('--quality', choices=['exact', 'lossless', 'high'], default='lossless')
    parser.add_argument('--diagnostic-profiler', action='store_true', help='Exclude stage-synchronized or instrumented server timings from primary evidence')
    args = parser.parse_args()
    out = Path('/task/outputs/benchmarks') / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    contract_path = args.contract
    contract = json.loads(contract_path.read_text())
    repo = args.repo or Path('/task/repos') / ('sglang' if args.arm == 'sgl' else 'vllm-omni')
    expected = 'af21e2b989e482cfdce39689126562f2209d53ac' if args.arm == 'sgl' else '817d4c17feb702289a2d2b2cbacfd594c262b7a7'
    sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert sha == (args.expected_sha or expected), sha
    patch = subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--binary', '--full-index'])
    patch_sha = hashlib.sha256(patch).hexdigest()
    if patch:
        assert args.expected_diff_sha256 == patch_sha, (args.expected_diff_sha256, patch_sha)
        (out / 'source.patch').write_bytes(patch)
    elif args.expected_diff_sha256:
        assert args.expected_diff_sha256 == patch_sha
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)
    (out / 'source-status.txt').write_text(status)
    nv.nvmlInit()
    handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in [0, 1]]
    allowed = set(args.allowed_gpu_pids)
    process_commands = {str(pid): Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode() for pid in allowed}
    assert all('vllm' in command or 'sgl_diffusion' in command for command in process_commands.values()), process_commands

    def sample():
        rows = []
        for index, handle in enumerate(handles):
            pids = {p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)}
            if pids - allowed:
                raise RuntimeError(f'Foreign GPU processes on GPU{index}: {pids - allowed}')
            mem = nv.nvmlDeviceGetMemoryInfo(handle)
            rows.append({'index': index, 'uuid': nv.nvmlDeviceGetUUID(handle), 'pids': sorted(pids),
                         'memory_bytes': mem.used, 'gpu_util_percent': nv.nvmlDeviceGetUtilizationRates(handle).gpu,
                         'power_mw': nv.nvmlDeviceGetPowerUsage(handle),
                         'temperature_c': nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
                         'sm_clock_mhz': nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)})
        return {'monotonic': time.monotonic(), 'gpus': rows}

    metadata = {'arm': args.arm, 'source_sha': sha, 'source_diff_sha256': patch_sha, 'source_repo': str(repo), 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'physical_gpus': [0], 'physical_gpu_pool': [0,1],
                'allowed_process_commands': process_commands, 'pid_namespace': 'container NVML-visible PIDs',
                'input_sha256': hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                'torch': importlib.metadata.version('torch'), 'driver': nv.nvmlSystemGetDriverVersion(),
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'initial_gpu_state': sample(), 'profiler': args.diagnostic_profiler,
                'timing_scope': 'localhost HTTP submission through full response bytes; PNG decoding/file I/O excluded',
                'lane': f'batch1 concurrency1 warm resident {args.configuration_label}; fixed prompt and seed; conditioning cache warm'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    base = 'http://127.0.0.1:' + ('31050' if args.arm == 'sgl' else '31051')
    image_data = None
    if contract['task'] == 'edit':
        image_path = Path('/task') / contract['reference_image']['path']
        image_data = image_path.read_bytes()
        assert hashlib.sha256(image_data).hexdigest() == contract['reference_image']['sha256']
    payload = {k: contract[k] for k in ['seed', 'num_inference_steps', 'guidance_scale', 'negative_prompt', 'generator_device']}
    payload.update(model=contract[args.arm + '_model_path'], prompt=next(p['prompt'] for p in contract['prompts'] if p['id']==args.prompt_id),
                   size=f'{contract["width"]}x{contract["height"]}', n=1, response_format='b64_json', output_format='png', output_compression=(0 if args.arm == 'sgl' else 100))
    if args.arm == 'sgl':
        payload.update(quality=args.quality, profile=False, max_sequence_length=contract['max_sequence_length'])
    results = []
    for i in range(args.warmups + args.repeats):
        directory = out / f'{i:02d}'
        directory.mkdir()
        request = dict(payload)
        if args.arm == 'sgl':
            request['perf_dump_path'] = str(directory / 'perf.json')
        if image_data is not None:
            request = {k: str(v).lower() if isinstance(v,bool) else str(v) for k,v in request.items()}
        (directory / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
        samples = [sample()]
        errors = []
        stop = threading.Event()

        def monitor():
            while not stop.wait(.25):
                try:
                    samples.append(sample())
                except Exception as exc:
                    errors.append(repr(exc))
                    return

        thread = threading.Thread(target=monitor)
        thread.start()
        start = time.monotonic()
        try:
            if image_data is None:
                response = requests.post(base + '/v1/images/generations', json=request, timeout=1200)
            else:
                response = requests.post(base + '/v1/images/edits', data=request,
                                         files={'image': ('reference.png', image_data, 'image/png')}, timeout=1200)
            elapsed = time.monotonic() - start
        finally:
            stop.set()
            thread.join()
            samples.append(sample())
            (directory / 'gpu-samples.json').write_text(json.dumps(samples) + '\n')
        if errors:
            raise RuntimeError(errors)
        (directory / 'response.json').write_text(response.text)
        response.raise_for_status()
        content = base64.b64decode(response.json()['data'][0]['b64_json'])
        (directory / 'output.png').write_bytes(content)
        row = {'index': i, 'warmup': i < args.warmups, 'http_seconds': elapsed,
               'sha256': hashlib.sha256(content).hexdigest(),
               'peak_gpu_memory_bytes': max(s['gpus'][0]['memory_bytes'] for s in samples)}
        results.append(row)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    measured = [r for r in results if not r['warmup']]
    values = [r['http_seconds'] for r in measured]
    half = len(values) // 2
    summary = {'n': len(values), 'mean_seconds': statistics.mean(values), 'p50_seconds': percentile(values, .5),
               'p95_seconds': percentile(values, .95), 'sample_stdev_seconds': statistics.stdev(values),
               'cv_percent': 100 * statistics.stdev(values) / statistics.mean(values),
               'sequential_requests_per_second': len(values) / sum(values),
               'aa_half_median_ratio': statistics.median(values[:half]) / statistics.median(values[half:]),
               'all_outputs_byte_identical': len({r['sha256'] for r in results}) == 1,
               'peak_gpu_memory_bytes': max(r['peak_gpu_memory_bytes'] for r in measured),
               'status': ('Diagnostic timing run (stage synchronization/instrumentation); excluded from primary E2E comparisons' if args.diagnostic_profiler else 'A/A evidence only; no candidate or cross-framework speedup claim')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    nv.nvmlShutdown()


if __name__ == '__main__':
    main()
