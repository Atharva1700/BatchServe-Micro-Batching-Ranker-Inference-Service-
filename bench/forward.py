"""Model-only cost: scoring rows at different batch sizes under torch.inference_mode, no HTTP.
Shows how much a batch of requests saves over one forward per request on this CPU."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batchserve.ranker import Ranker, item_table  # noqa: E402

torch.set_num_threads(1)
CANDS = 100
model = Ranker().eval()
table = item_table()
rng = np.random.default_rng(0)
out = {}
for B in [1, 4, 16, 64]:
    ids = rng.integers(0, len(table), (B, CANDS))
    feats = torch.from_numpy(table[ids.reshape(-1)])
    reps = max(20, 2000 // B)
    with torch.inference_mode():
        model(feats[:CANDS])
        t0 = time.perf_counter()
        for _ in range(reps):
            model(feats)
        per_call = (time.perf_counter() - t0) / reps
    per_req_ms = per_call / B * 1000
    out[B] = {"ms_per_call": round(per_call * 1000, 3), "ms_per_request": round(per_req_ms, 4),
              "requests_per_s": round(B / per_call)}
    print(B, out[B], flush=True)
(Path(__file__).resolve().parent.parent / "results" / "forward.json").write_text(
    json.dumps({"candidates_per_request": CANDS, "torch_threads": 1, "results": out}, indent=2))
