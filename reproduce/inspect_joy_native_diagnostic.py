"""Correlate native profiler log entries, complete observer rows and HTTP responses."""
import hashlib
import json
import re
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def inspect(root, shape, archived=False):
    plan = read(root / ('benchmarks' if archived else '') / f'joy-gelu-final-{shape}-omni-denoise1-plan.json')
    name = plan['id'] + '-00-omni_compile_profile'
    outputs = root if archived else root / 'outputs'
    folder = outputs / 'benchmarks' / name
    server = root / 'block-runs' / name
    raw_log = (server / 'server.log').read_bytes()
    log = re.sub(r'\x1b\[[0-9;]*m', '', raw_log.decode(errors='replace'))
    matches = list(re.finditer(r'\(DiffusionWorker pid=(\d+)\).*?\[DiffusionPipelineProfiler\] JoyImageEditPipeline\.diffuse took ([0-9.]+)s', log))
    count = 4 + plan['warmups'] + plan['repeats']
    assert len(matches) == count + 1, 'Retain and inspect any unexpected profiler calls'
    pids = {int(match[1]) for match in matches}
    assert len(pids) == 1
    worker = pids.pop()
    precision_path = (folder / 'runtime-precision.jsonl' if archived else
                      outputs / 'precision/omni-native' / f'omni-worker-{worker}.jsonl')
    raw_precision = precision_path.read_bytes()
    precision = [json.loads(line) for line in raw_precision.splitlines()]
    assert len(precision) == count + 1
    assert len(precision[0]['scheduler_calls']) == 2
    assert all(len(row['scheduler_calls']) == 30 for row in precision[1:])
    assert matches[0].end() < log.index('Application startup complete.') < matches[1].start()
    assert all(row['pid'] == worker and row['name'] == plan['required_stage_metric']
               and row['hook_sha256'] == plan['precision_observer']['sha256'] for row in precision)
    assert all(a['finished_ns'] < b['started_ns'] for a,b in zip(precision, precision[1:]))
    quality_folder = outputs / 'qualification' / ('omni-joy-' + name)
    quality = read(quality_folder / 'results.json')
    records = read(folder / 'results.json')
    assert len(quality) == 4 and len(records) == plan['warmups'] + plan['repeats']
    requests = [(quality_folder / Path(row['request']).name, 'qualification', row)
                for row in quality]
    requests += [(folder / f'{row["index"]:02d}', 'warmup' if row['warmup'] else 'diagnostic', row)
                 for row in records]
    correlations = [dict(observer_index=0, phase='native_startup_warmup', scheduler_steps=2,
        native_diffuse_seconds=float(matches[0][2]), started_ns=precision[0]['started_ns'],
        finished_ns=precision[0]['finished_ns'], excluded_from_request_metrics=True)]
    values = []
    for index, (request_folder, phase, row) in enumerate(requests, 1):
        observed = precision[index]
        response = read(request_folder / 'response.json')
        value = float(matches[index][2])
        assert 0 < value < row['http_seconds']
        # API created is integer seconds, set after generation including VAE.
        # Match every response to the corresponding completed diffuse call.
        gap = response['created'] - observed['finished_ns'] / 1e9
        assert -1 <= gap <= 5, (index, gap)
        assert (observed['finished_ns'] - observed['started_ns']) / 1e9 >= value - .001
        if phase == 'diagnostic':
            values.append(value)
        correlations.append(dict(observer_index=index, phase=phase,
            request=str(request_folder), response_created=response['created'],
            response_created_minus_observer_finish_seconds=gap,
            output_sha256=row['sha256'], scheduler_steps=len(observed['scheduler_calls']),
            native_diffuse_seconds=value, started_ns=observed['started_ns'], finished_ns=observed['finished_ns']))
    assert len(values) == plan['repeats']
    return dict(shape=shape, primary_timing=False, all_calls_accounted_for=True,
        metric=plan['required_stage_metric'], metric_source='Native synchronized profiler server.log; six decimal seconds. HTTP exposes only queue/stage aggregate metrics.',
        observer_records=count+1, native_startup_records=1, correlated_request_records=count,
        denoise_seconds=values, denoise_p50_seconds=statistics.median(values),
        worker_pid=worker, server_log_sha256=hashlib.sha256(raw_log).hexdigest(),
        precision_jsonl_sha256=hashlib.sha256(raw_precision).hexdigest(),
        correlations=correlations), raw_precision
