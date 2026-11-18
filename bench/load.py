"""Open-loop load test: requests are sent on a fixed schedule regardless of responses. Latency is measured
from the scheduled send time, so client-side lag is included. Each config runs in a fresh uvicorn server.
Usage: python bench/load.py [duration_s]"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PORT = 8089
CONFIGS = {
    "unbatched": {"BATCH_MAX": "1", "BATCH_WAIT_MS": "0", "DEADLINE_MS": "50"},
    "batched_64_5ms": {"BATCH_MAX": "64", "BATCH_WAIT_MS": "5", "DEADLINE_MS": "50"},
}
LEVELS = [200, 500, 800, 1200, 1600, 2400, 3200]


def start_server(env_extra):
    env = {**os.environ, **env_extra}
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "batchserve.app:app", "--port", str(PORT),
                          "--log-level", "warning"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for _ in range(200):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/healthz", timeout=0.5).status_code == 200:
                return p
        except httpx.HTTPError:
            time.sleep(0.1)
    p.kill()
    raise RuntimeError("server did not start")


async def drive(rps, dur, rng, conns=256):
    """Open loop with a raw HTTP/1.1 keep-alive client (httpx used too much of the 2 cores to be a fair client).
    A producer enqueues each request at its scheduled time; `conns` workers take them in order."""
    n = int(rps * dur)
    bodies = [json.dumps({"candidates": c, "top_k": 10}).encode()
              for c in rng.integers(0, 50_000, (n, 100)).tolist()]
    head = (b"POST /rank HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n")
    q: asyncio.Queue = asyncio.Queue()
    lat, codes, srcs = [], {}, {}

    async def worker():
        reader, writer = await asyncio.open_connection("127.0.0.1", PORT)
        while True:
            item = await q.get()
            if item is None:
                break
            i, sched = item
            try:
                writer.write(head % len(bodies[i]) + bodies[i])
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), 10)
                status = int(line.split()[1])
                clen = 0
                while True:
                    h = await reader.readline()
                    if h in (b"\r\n", b""):
                        break
                    if h.lower().startswith(b"content-length:"):
                        clen = int(h.split(b":")[1])
                payload = await reader.readexactly(clen)
                codes[status] = codes.get(status, 0) + 1
                if status == 200:
                    s = "fallback" if b'"fallback"' in payload else "model"
                    srcs[s] = srcs.get(s, 0) + 1
            except (asyncio.TimeoutError, OSError, ValueError, IndexError):
                codes["error"] = codes.get("error", 0) + 1
                writer.close()
                reader, writer = await asyncio.open_connection("127.0.0.1", PORT)
            lat.append(time.perf_counter() - sched)
        writer.close()

    workers = [asyncio.create_task(worker()) for _ in range(conns)]
    t0 = time.perf_counter()
    for i in range(n):
        sched = t0 + i / rps
        d = sched - time.perf_counter()
        if d > 0:
            await asyncio.sleep(d)
        await q.put((i, sched))
    for _ in workers:
        await q.put(None)
    await asyncio.gather(*workers)
    wall = time.perf_counter() - t0
    lat_ms = np.array(lat) * 1000
    ok = codes.get(200, 0)
    return {
        "offered_rps": rps, "requests": n, "achieved_rps": round(ok / wall, 1),
        "p50_ms": round(float(np.percentile(lat_ms, 50)), 2),
        "p99_ms": round(float(np.percentile(lat_ms, 99)), 2),
        "status": {str(k): v for k, v in codes.items()},
        "fallback_frac": round(srcs.get("fallback", 0) / max(ok, 1), 4),
    }


def server_stats():
    return httpx.get(f"http://127.0.0.1:{PORT}/stats", timeout=5).json()


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    results = {}
    for name, env in CONFIGS.items():
        results[name] = []
        for rps in LEVELS:
            srv = start_server(env)
            try:
                before = server_stats()
                row = asyncio.run(drive(rps, dur, np.random.default_rng(rps)))
                after = server_stats()
            finally:
                srv.kill()
                srv.wait()
            batches = after["batches"] - before["batches"]
            served = (after["model"] - before["model"]) + (after["fallback"] - before["fallback"])
            hist = {k: after["batch_hist"].get(k, 0) - before["batch_hist"].get(k, 0)
                    for k in after["batch_hist"]}
            hist = {k: v for k, v in hist.items() if v > 0}
            row["batches"] = batches
            row["mean_batch"] = round(served / batches, 2) if batches else 0.0
            row["batch_hist"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
            results[name].append(row)
            print(name, json.dumps(row), flush=True)
    out = ROOT / "results" / "load.json"
    out.write_text(json.dumps({"duration_s": dur, "candidates_per_request": 100, "deadline_ms": 50,
                               "results": results}, indent=2))


if __name__ == "__main__":
    main()
