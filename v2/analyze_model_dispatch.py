import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = {arm: json.loads((root / f"model-{arm}.json").read_text())
        for arm in ("A1", "B1", "B2", "A2")}
summary = []
for i, case in enumerate(runs["A1"]):
    values = {arm: rows[i]["eager_us"] for arm, rows in runs.items()}
    assert all(case["name"] == rows[i]["name"] and
               case["output"] == rows[i]["output"] and rows[i]["verified"]
               for rows in runs.values())
    baseline = statistics.mean([values["A1"], values["A2"]])
    candidate = statistics.mean([values["B1"], values["B2"]])
    summary.append(dict(name=case["name"], baseline_us=baseline,
                        candidate_us=candidate, delta_pct=(candidate / baseline - 1) * 100,
                        runs_us=values, all_outputs_equal=True, all_gates_verified=True))
print(json.dumps(summary, indent=2))
