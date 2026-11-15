"""Two outstanding native jobs per group; output/state admission, not timing."""
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

from build_nomas_video_payload import build_payload

arm, run = sys.argv[1:]
assert arm in ("sgl", "omni") and Path(run).name == run
folder = Path("/task/runs") / run
visit = json.loads((folder / "visit.json").read_text())
contract = json.loads(Path("/task/inputs/nomas-native-workloads01.json").read_text())
assert visit["concurrency"] == 2 and len(visit["requests"]) == 12
out = folder / "outputs"
out.mkdir(exist_ok=False)
base = "http://127.0.0.1:" + ("32150" if arm == "sgl" else "32151")
terminal = {"completed", "failed", "cancelled"}
rows = []
with requests.Session() as session:
    session.trust_env = False
    for group in range(6):
        start_wall_ns = time.time_ns()
        jobs = []
        for index in range(group * 2, group * 2 + 2):
            item = visit["requests"][index]
            directory = out / f"{index:02d}-{item['prompt_id']}"
            directory.mkdir()
            payload = build_payload(arm, contract, item, directory)
            (directory / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
            sent = time.time_ns()
            response = session.post(base + "/v1/videos", timeout=120,
                                    **({"json": payload} if arm == "sgl" else {"data": payload}))
            (directory / "submit-response.txt").write_text(response.text)
            response.raise_for_status()
            job = response.json()
            assert job["status"] not in terminal, "POST must return while native job remains outstanding"
            jobs.append(dict(index=index, item=item, directory=directory, job=job,
                             submit_wall_ns=sent, submitted_wall_ns=time.time_ns()))
        # Observe both after the second POST: overlapping client requests are real.
        snapshot = []
        for entry in jobs:
            response = session.get(base + "/v1/videos/" + entry["job"]["id"], timeout=120)
            response.raise_for_status()
            entry["job"] = response.json()
            snapshot.append(entry["job"])
        (out / f"group-{group}-concurrent-status.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        assert all(job["status"] not in terminal for job in snapshot), snapshot
        deadline = time.monotonic() + 7200
        while any(entry["job"]["status"] not in terminal for entry in jobs):
            assert time.monotonic() < deadline
            time.sleep(visit["poll_seconds"])
            for entry in jobs:
                if entry["job"]["status"] in terminal:
                    continue
                response = session.get(base + "/v1/videos/" + entry["job"]["id"], timeout=120)
                response.raise_for_status()
                entry["job"] = response.json()
                with (entry["directory"] / "poll.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(wall_ns=time.time_ns(), job=entry["job"])) + "\n")
        end_wall_ns = time.time_ns()
        events = [json.loads(line) for p in (folder / "runtime-audit").glob("*.jsonl")
                  for line in p.read_text().splitlines()]
        events = [e for e in events if start_wall_ns <= e["wall_ns"] <= end_wall_ns]
        begins = [e for e in events if e["event"] == "request_begin"]
        assert len(begins) == 2, begins
        for entry in jobs:
            index, item, directory, job = (entry[k] for k in ("index", "item", "directory", "job"))
            (directory / "terminal-response.json").write_text(json.dumps(job, indent=2) + "\n")
            assert job["status"] == "completed", job
            prompt_hash = hashlib.sha256(contract["prompts"][item["prompt_id"]].encode()).hexdigest()
            matches = [e for e in begins if e["seed"] == item["seed"] and e["prompt_sha256"] == prompt_hash]
            assert len(matches) == 1, matches
            begin = matches[0]
            own = [e for e in events if (e["pid"], e["request"]) == (begin["pid"], begin["request"])]
            predictions = [e for e in own if e["event"] == "native_predict" and not e["commit"]]
            commits = [e for e in own if e["event"] == "native_predict" and e["commit"]]
            inputs = [e for e in own if e["event"] == "native_inputs"]
            complete = [e for e in own if e["event"] == "denoise_complete"]
            assert len(inputs) == len(complete) == 1 and len(predictions) == item["steps"] == 8
            assert inputs[0]["tensors"]["initial_noise"]["shape"] == [1, 32, (item["frames"] - 1) // 4 + 1, item["height"] // 32, item["width"] // 32]
            if item.get("expected_mean_square_calls"):
                fusions = [e for e in own if e["event"] == "vae_mean_square_fusion"]
                assert len(fusions) == 1 and fusions[0]["verified"] and not fusions[0]["gate_disabled"]
                assert sum(c["count"] for c in fusions[0]["calls"]) == item["expected_mean_square_calls"]
            (directory / "native-audit.json").write_text(json.dumps(own, indent=2) + "\n")
            response = session.get(base + "/v1/videos/" + job["id"] + "/content", timeout=120)
            response.raise_for_status()
            assert "video/mp4" in response.headers.get("Content-Type", "")
            content = response.content
            (directory / "output.mp4").write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            expected = item["expected_sha256"][arm]
            row = dict(index=index, prompt_id=item["prompt_id"], seed=item["seed"], frames=item["frames"],
                       workload_id=item["id"], role="admission", benchmark=False, profiler=False,
                       concurrency=2, actual_overlapping_jobs=True, native_steps_observed=len(predictions),
                       native_commit_forwards=len(commits), sha256=digest, expected_sha256=expected,
                       matches_reference=digest == expected, output=str(directory / "output.mp4"),
                       bytes=len(content), native_request=begin["request"], native_pid=begin["pid"],
                       group_start_wall_ns=start_wall_ns, group_end_wall_ns=end_wall_ns,
                       submit_wall_ns=entry["submit_wall_ns"], submitted_wall_ns=entry["submitted_wall_ns"])
            rows.append(row)
            (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            assert digest == expected, "C2 differs from same-arm C1; preserve and investigate"
