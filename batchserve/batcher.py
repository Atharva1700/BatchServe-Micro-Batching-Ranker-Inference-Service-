"""Micro-batcher: requests queue up and are flushed as one forward pass when either `max_batch` requests
are queued or `max_wait_ms` has passed since the first queued request. A request whose deadline has passed
when its batch is flushed is answered from the popularity fallback and is not scored."""
import asyncio
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from batchserve.ranker import Ranker, item_table, popularity


@dataclass
class _Pending:
    ids: np.ndarray
    deadline: float
    fut: asyncio.Future


class MicroBatcher:
    def __init__(self, max_batch=64, max_wait_ms=5.0, deadline_ms=50.0, seed=0):
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.deadline_s = deadline_ms / 1000
        self.model = Ranker(seed).eval()
        self.table = item_table(seed)
        self.pop = popularity(seed)
        self.q: asyncio.Queue = asyncio.Queue()
        self.stats = {"batches": 0, "batch_hist": Counter(), "model": 0, "fallback": 0}
        self._task = None

    async def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._task.cancel()

    async def score(self, ids: np.ndarray):
        """Waits at most deadline_s. On timeout the future is cancelled (the batcher then skips it)
        and the caller gets the popularity fallback immediately, so latency is bounded by the deadline."""
        fut = asyncio.get_running_loop().create_future()
        await self.q.put(_Pending(ids, time.monotonic() + self.deadline_s, fut))
        try:
            return await asyncio.wait_for(fut, self.deadline_s)
        except asyncio.TimeoutError:
            self.stats["fallback"] += 1
            return self.pop[ids], "fallback"

    async def _loop(self):
        while True:
            batch = [await self.q.get()]
            flush_at = time.monotonic() + self.max_wait
            while len(batch) < self.max_batch:
                left = flush_at - time.monotonic()
                if left <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.q.get(), left))
                except asyncio.TimeoutError:
                    break
            self._run(batch)

    def _run(self, batch):
        now = time.monotonic()
        live = []
        for p in batch:
            if p.fut.done():
                continue
            if now > p.deadline:
                continue         # expired: score() times out on its own and answers with the fallback
            else:
                live.append(p)
        self.stats["batches"] += 1
        self.stats["batch_hist"][len(batch)] += 1
        if not live:
            return
        feats = torch.from_numpy(np.concatenate([self.table[p.ids] for p in live]))
        with torch.inference_mode():
            scores = self.model(feats).numpy()
        off = 0
        for p in live:
            k = len(p.ids)
            if not p.fut.done():
                p.fut.set_result((scores[off:off + k], "model"))
                self.stats["model"] += 1
            off += k
