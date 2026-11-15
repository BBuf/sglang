"""Complete queued-concurrency admission after the owned SP2 visit finishes."""
import datetime
import json
import subprocess
import time
from pathlib import Path

root = Path("/data/xyzhang/diffusion-6000-20261113")
path = root / "evidence/preflight/nomas-c2_23-controller.json"
assert not path.exists()
state = dict(status="waiting_sp2_18", benchmark=False, completed=[], physical_gpus=[0])


def save(**updates):
    state.update(updates, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state, indent=2) + "\n")


try:
    save()
    deadline = time.monotonic() + 3 * 3600
    while True:
        previous = json.loads((root / "evidence/preflight/nomas-sp2_18-controller.json").read_text())
        if previous["status"] == "complete" or previous["status"].startswith("failed"):
            break
        assert time.monotonic() < deadline, "prior visit deadline; inspect without restart"
        time.sleep(15)
    save(prior_SP2_terminal_state=previous)
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    for label in ["sgl", "omni", "PR"]:
        arm = "omni" if label == "omni" else "sgl"
        name = f"nomas-c2-{label}-native23"
        command = ["python3", str(root / "scripts/nomas-c2-visits23/run.py"),
                   arm, name, f"inputs/{name}.json"]
        save(status="running", active=name, command=command)
        subprocess.run(command, check=True)
        folder = root / "runs" / name
        status = json.loads((folder / "status.json").read_text())
        assert status["status"] == "raw_complete" and status["gpu_cleanup"] == "idle_verified", status
        rows = json.loads((folder / "outputs/results.json").read_text())
        assert len(rows) == 6 and all(x["matches_reference"] and x["actual_overlapping_jobs"] for x in rows)
        assert rows[0]["sha256"] == rows[4]["sha256"] and rows[1]["sha256"] == rows[5]["sha256"]
        state["completed"].append(name)
        save()
    save(status="complete", active=None,
         quality_scope="All C2 outputs exact to actual same-arm C1 outputs; existing cross-framework failures remain.")
except BaseException as exc:
    save(status="failed_preserve_no_automatic_retry", error=repr(exc))
    raise
