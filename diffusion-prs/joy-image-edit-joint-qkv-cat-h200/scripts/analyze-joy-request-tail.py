"""Align diagnostic request-tail and GC intervals without treating them as benchmarks."""
import json
from pathlib import Path
import re
import statistics
import time

root = Path('/campaign')
done = root / 'diagnose-joy-request-tail-retry2.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
decoder = json.JSONDecoder()
report = {}
for arm in ['baseline', 'candidate']:
    log = (root / f'joy-image-edit-tail-diagnostic-{arm}.log').read_text()
    active = -1
    rows = {}
    startup = []
    for match in re.finditer(r'CAMPAIGN_(REQUEST_START|REQUEST_END|TAIL|GC) ', log):
        kind = match.group(1)
        rest = log[match.end():]
        if kind == 'REQUEST_START':
            active = int(rest.split()[0])
            rows[active] = dict(tail=[], gc=[])
            continue
        data, _ = decoder.raw_decode(rest)
        if active < 0:
            startup.append(dict(kind=kind, data=data))
            continue
        if kind == 'REQUEST_END':
            assert data['index'] == active
            rows[active]['request'] = data
        else:
            rows[active][kind.lower()].append(data)
    assert len(rows) == 12
    for index, row in rows.items():
        request = row['request']
        row['saved_client_minus_worker_s'] = request['client_wall_s'] - request['worker_s']
        row['largest_timed_tail'] = max(row['tail'], key=lambda x:x['seconds'])
        row['gc_total_s'] = sum(x['seconds'] for x in row['gc'])
    components = {}
    for name in sorted({x['name'] for r in rows.values() for x in r['tail']}):
        components[name] = {}
        for rank in ['0', '1', None]:
            values = [x['seconds'] for r in rows.values() for x in r['tail'] if x['name'] == name and x['rank'] == rank]
            if values:
                components[name][str(rank)] = dict(count=len(values), median_s=statistics.median(values), max_s=max(values))
    report[arm] = dict(rows=rows, components=components, startup=startup,
        source=(root / f'artifacts/joy-image-edit/persistent/{arm}/source.txt').read_text().splitlines()[0],
        parent='80da4432d085ed4d6166ef643d9fd2b829dbb0c5' if arm=='baseline' else '4811f4d52aa2586412f699b3bb84ed184d76250a')
out = root / 'artifacts/joy-image-edit/request-tail-diagnostic.json'
out.write_text(json.dumps(report, indent=2))
print(json.dumps({a:dict(components=r['components'], requests=[dict(index=i,
    worker_s=x['request']['worker_s'], client_s=x['request']['client_wall_s'], tail=x['saved_client_minus_worker_s'],
    largest=x['largest_timed_tail'], gc_s=x['gc_total_s']) for i,x in r['rows'].items()]) for a,r in report.items()}, indent=2), flush=True)
