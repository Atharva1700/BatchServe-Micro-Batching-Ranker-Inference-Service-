"""FastAPI front end. Config from env: BATCH_MAX (1 = unbatched), BATCH_WAIT_MS, DEADLINE_MS."""
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from batchserve.batcher import MicroBatcher
from batchserve.ranker import N_ITEMS


class RankRequest(BaseModel):
    candidates: list[int] = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=500)


def create_app(max_batch=64, max_wait_ms=5.0, deadline_ms=50.0):
    batcher = MicroBatcher(max_batch, max_wait_ms, deadline_ms)

    @asynccontextmanager
    async def lifespan(app):
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.batcher = batcher

    @app.post("/rank")
    async def rank(req: RankRequest):
        ids = np.asarray(req.candidates, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= N_ITEMS:
            from fastapi import HTTPException
            raise HTTPException(422, f"candidate ids must be in [0, {N_ITEMS})")
        scores, source = await batcher.score(ids)
        order = np.argsort(-scores, kind="stable")[: req.top_k]
        return {"ranked": ids[order].tolist(), "source": source}

    @app.get("/stats")
    def stats():
        s = batcher.stats
        return {"batches": s["batches"], "model": s["model"], "fallback": s["fallback"],
                "batch_hist": {str(k): v for k, v in sorted(s["batch_hist"].items())}}

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


app = create_app(int(os.environ.get("BATCH_MAX", 64)),
                 float(os.environ.get("BATCH_WAIT_MS", 5.0)),
                 float(os.environ.get("DEADLINE_MS", 50.0)))
