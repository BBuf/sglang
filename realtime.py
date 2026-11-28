"""Replay the recorded two-segment requests and verify every delivered RGB frame."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import msgspec.msgpack
import websockets


async def replay(record, port):
    protocol = record["protocol"]
    width, height = map(int, protocol["init"]["size"].split("x"))
    stride = width * height * 3
    chunks, digests, session_ids = [], [], set()
    endpoint = f"ws://127.0.0.1:{port}/v1/realtime_video/generate"
    async with websockets.connect(endpoint, max_size=None, ping_interval=None) as ws:
        await ws.send(msgspec.msgpack.encode(protocol["init"]))
        async for message in ws:
            header = msgspec.msgpack.decode(message)
            kind = header["type"]
            if kind == "error":
                raise RuntimeError(header)
            if kind == "chunk_stats":
                index = header["chunk_index"]
                if index != len(chunks) or not header["session_id"]:
                    raise ValueError(header)
                session_ids.add(header["session_id"])
                chunks.append(dict(chunk=index, frames=header["num_frames"], raw_bytes=header["raw_bytes"]))
                event = protocol.get(f"after_chunk_{index}")
                if event:
                    await ws.send(msgspec.msgpack.encode(event))
                continue
            if kind not in ("frame_batch", "frame_batch_header"):
                raise ValueError(header)
            payload = header["payload"] if kind == "frame_batch" else await ws.recv()
            count = header["num_frames"]
            if not isinstance(payload, bytes) or len(payload) != count * stride:
                raise ValueError("Invalid RGB frame payload")
            if count and (header["width"],header["height"],header["channels"],header["format"]) != (width,height,3,"rgb24"):
                raise ValueError("Unexpected frame geometry or format")
            view = memoryview(payload)
            digests.extend(hashlib.sha256(view[p:p+stride]).hexdigest() for p in range(0,len(payload),stride))
    if chunks != record["chunks"] or digests != record["frames"] or len(session_ids) != 1:
        raise ValueError("Realtime delivery differs from the recorded request")
    return dict(seed=record["case"]["seed"], frames=len(digests), frame_sha256=digests, chunks=chunks,
                session_id=next(iter(session_ids)))


async def run(port, output):
    evidence = json.loads(Path(__file__).with_name("realtime-session-controls.json").read_text())
    series = evidence["head"]
    if [r["frames"] for r in series] != [r["frames"] for r in evidence["main"]]:
        raise ValueError("The main and head receipts disagree")
    results = []
    for request in series:
        results.append(await asyncio.wait_for(replay(request, port), timeout=480))
    if len({r["session_id"] for r in results}) != len(results):
        raise ValueError("The server reused a realtime session ID")
    with output.open("x") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"{len(results)} sessions and {sum(r['frames'] for r in results)} exact RGB frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=32960)
    parser.add_argument("--out", type=Path, default=Path("realtime-results.json"))
    options = parser.parse_args()
    asyncio.run(run(options.port, options.out))
