"""Self-checks. Run: python -m pytest -q tests  (or python tests/test_batchserve.py)"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batchserve.app import create_app  # noqa: E402
from batchserve.batcher import MicroBatcher  # noqa: E402


async def _fire(b, n, k=20):
    rng = np.random.default_rng(0)
    return await asyncio.gather(*[b.score(rng.integers(0, 1000, k)) for _ in range(n)])


def test_flush_on_size_before_timer():
    async def go():
        b = MicroBatcher(max_batch=4, max_wait_ms=2000)
        await b.start()
        t0 = time.monotonic()
        await _fire(b, 4)
        dt = time.monotonic() - t0
        await b.stop()
        assert b.stats["batch_hist"] == {4: 1}
        assert dt < 1.0
    asyncio.run(go())


def test_flush_on_timer_when_under_size():
    async def go():
        b = MicroBatcher(max_batch=64, max_wait_ms=5)
        await b.start()
        await _fire(b, 2)
        await b.stop()
        assert b.stats["batch_hist"] == {2: 1}
    asyncio.run(go())


def test_expired_requests_fall_back_without_scoring():
    async def go():
        b = MicroBatcher(max_batch=8, max_wait_ms=5, deadline_ms=0)
        await b.start()
        out = await _fire(b, 5)
        await b.stop()
        assert all(src == "fallback" for _, src in out)
        assert b.stats["model"] == 0 and b.stats["fallback"] == 5
    asyncio.run(go())


def test_batched_scores_equal_unbatched():
    async def go():
        ids = np.arange(0, 100)
        b1 = MicroBatcher(max_batch=1, max_wait_ms=0, deadline_ms=5000)
        b8 = MicroBatcher(max_batch=8, max_wait_ms=50, deadline_ms=5000)
        await b1.start(); await b8.start()
        solo = [(await b1.score(ids[i * 10:(i + 1) * 10]))[0] for i in range(8)]
        batched = await asyncio.gather(*[b8.score(ids[i * 10:(i + 1) * 10]) for i in range(8)])
        await b1.stop(); await b8.stop()
        for s, (bs, _) in zip(solo, batched):
            np.testing.assert_allclose(s, bs, rtol=1e-5, atol=1e-6)
        assert b8.stats["batch_hist"] == {8: 1}
    asyncio.run(go())


def test_endpoint_roundtrip_and_validation():
    app = create_app(max_batch=16, max_wait_ms=5, deadline_ms=5000)
    with TestClient(app) as c:
        r = c.post("/rank", json={"candidates": [3, 9, 27, 81], "top_k": 2})
        assert r.status_code == 200
        body = r.json()
        assert len(body["ranked"]) == 2 and body["source"] == "model"
        assert c.post("/rank", json={"candidates": []}).status_code == 422
        assert c.post("/rank", json={"candidates": [10**9]}).status_code == 422


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
