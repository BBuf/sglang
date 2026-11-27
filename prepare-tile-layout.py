"""Create a local view of MiniMax-H3-Image with its native distributed tile switch enabled."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(checkpoint, output):
    checkpoint = checkpoint.resolve(strict=True)
    output = output.resolve()
    source = checkpoint / "vae" / "config.json"
    source_bytes = source.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == "7fcc21bec549804ff0a04d8246e7f90e679d4f8d4b0a27a252bbd18c30e3e4bb"
    config = json.loads(source_bytes)
    assert config["vae_parallel_tiling"] == 0
    config["vae_parallel_tiling"] = 1
    output.mkdir()
    for entry in checkpoint.iterdir():
        target = output / entry.name
        if entry.name == "vae":
            target.mkdir()
            for component in entry.iterdir():
                if component.name == "config.json":
                    (target / component.name).write_text(json.dumps(config, indent=2) + "\n")
                else:
                    (target / component.name).symlink_to(component.resolve())
        else:
            target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
    assert (output / "vae" / "diffusion_pytorch_model.safetensors").samefile(checkpoint / "vae" / "diffusion_pytorch_model.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.checkpoint, args.output)
