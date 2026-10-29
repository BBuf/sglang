"""Fetch only pinned model metadata and small JSON configs; never weights."""
import concurrent.futures
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "480p": "db5f398b13ca086d09a50ce156c20527773841b1",
    "720p": "82de0b2398cd16435d3f6621778f9e89ffa096cb",
}
DEST = ROOT / "evidence/downloads/sana-video-pinned-metadata"
DEST.mkdir(exist_ok=True)


def fetch(variant, revision):
    model = f"Efficient-Large-Model/SANA-Video_2B_{variant}_diffusers"
    files = {
        "hf-model.json": f"https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true",
        **{
            name: f"https://huggingface.co/{model}/resolve/{revision}/{name}"
            for name in [
                "model_index.json", "transformer/config.json", "vae/config.json",
                "scheduler/scheduler_config.json", "text_encoder/config.json",
                "tokenizer/tokenizer_config.json",
            ]
        },
    }
    out = []
    for name, url in files.items():
        target = DEST / variant / name
        assert not target.exists(), target
        request = urllib.request.Request(url, headers={"User-Agent": "Codex-metadata-audit"})
        with urllib.request.urlopen(request, timeout=40) as response:
            data = response.read(2_000_001)
            assert len(data) <= 2_000_000
            value = json.loads(data)
            if name == "hf-model.json":
                assert value["sha"] == revision
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            out.append(dict(file=name, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return dict(model=model, revision=revision, files=out)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(lambda pair: fetch(*pair), MODELS.items()))
    receipt = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), weights_downloaded=False, records=records)
    (DEST / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
