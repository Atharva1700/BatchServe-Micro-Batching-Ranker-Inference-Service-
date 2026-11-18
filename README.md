# BatchServe: Micro-Batching Ranker Inference Service

An asyncio micro-batcher in front of a PyTorch ranker, exposed through FastAPI. Concurrent `/rank` requests
are grouped into one forward pass when either 64 requests are queued or 5 ms has passed since the first one.
A request whose 50 ms deadline expires is answered from a popularity fallback and is not scored.

## Layout

- `batchserve/ranker.py`: MLP ranker (32 → 128 → 64 → 1), fixed seeded random weights, item feature table, popularity prior.
- `batchserve/batcher.py`: `MicroBatcher`. Queue, timer-or-size flush, deadline handling, `torch.inference_mode` scoring.
- `batchserve/app.py`: FastAPI app. `POST /rank`, `GET /stats`, `GET /healthz`. Config from env:
  `BATCH_MAX` (1 = unbatched), `BATCH_WAIT_MS`, `DEADLINE_MS`.
- `bench/forward.py`: model-only cost at batch sizes 1, 4, 16, 64 → `results/forward.json`.
- `bench/load.py`: open-loop load test, unbatched vs batched, 7 offered rates → `results/load.json` (log in `results/load.log`).
- `tests/test_batchserve.py`: five self-checks.

Run: `python -m pytest -q tests`. Serve: `uvicorn batchserve.app:app`. Load test: `python bench/load.py 6` (about 5 min).

## Behaviour

- **Flush.** The loop takes the first queued request, then keeps collecting until 64 requests are queued or
  `max_wait` has passed. All live requests in the batch are concatenated into one `(rows, 32)` tensor.
- **Deadline and shedding.** `score()` waits at most `DEADLINE_MS`. On timeout it cancels its own queue entry
  and returns the popularity scores for its candidates (`"source": "fallback"`). Entries that expire before
  their batch is flushed are skipped, so no model work is spent on them.
- **Unbatched.** `BATCH_MAX=1` with a 0 ms wait gives one forward pass per request. The same code path is used.

## Results (2 CPU cores, no GPU)

Model-only cost (`results/forward.json`, 100 candidates per request, one torch thread):

| Requests per forward | ms per call | ms per request | requests/s |
|---|---|---|---|
| 1 | 0.057 | 0.057 | 17,465 |
| 4 | 0.119 | 0.030 | 33,642 |
| 16 | 0.441 | 0.028 | 36,296 |
| 64 | 2.026 | 0.032 | 31,585 |

Batching cuts per-request model cost by about half, but the model is a small fraction of the request path.
Most of the cost is HTTP and JSON handling, which is why the service-level numbers below are close.

Open-loop load, 6 s per level, 100 candidates per request, deadline 50 ms
(`model_rps` = achieved responses minus fallback responses):

| Offered rps | Unbatched p50 / p99 (ms) | Unbatched model rps | Batched p50 / p99 (ms) | Batched model rps | Batched fallback | Batched mean batch |
|---|---|---|---|---|---|---|
| 200 | 1.8 / 7.3 | 200 | 7.3 / 25.6 | 199 | 0% | 1.7 |
| 500 | 1.7 / 15.6 | 498 | 6.6 / 34.8 | 498 | 0% | 3.5 |
| 800 | 1.8 / 38.8 | 798 | 7.3 / 27.0 | 796 | 0% | 5.6 |
| 1200 | 4.3 / 36.4 | 1197 | 8.3 / 47.4 | 1196 | 0% | 8.5 |
| 1600 | 13.7 / 73.2 | 1596 | 13.7 / 63.7 | 1596 | 0% | 11.7 |
| 2400 | 493 / 923 | 2053 | 427 / 540 | 2102 | 5.3% | 38.6 |
| 3200 | 1874 / 3581 | 1975 | 1070 / 2055 | 2241 | 6.0% | 42.9 |

What the numbers show:
- **Light load (≤1600 rps): unbatched is faster.** Batching adds about 5 ms of waiting for the flush window,
  which is most of the 1.8 → 7.3 ms p50 difference at 200 rps.
- **Overload (2400–3200 rps): batched serves more.** Model-served throughput is about 2–13% higher, and p99 is
  lower at 2400 rps (540 vs 923 ms). The deadline sheds about 5–6% of requests to the fallback. Without
  shedding, the queue would keep growing.
- **The deadline does not bound end-to-end latency under saturation.** At 2400–3200 rps the server event loop
  is itself saturated, so p50 stays in the hundreds of milliseconds even though the batcher answers within
  50 ms. A production fix would be admission control at the HTTP layer (reject or shed before parsing the body).
  This was not built here.

## Tests

- Flush on size happens before the timer (4 requests, one batch, under 1 s with a 2 s timer).
- Flush on timer happens when fewer requests than the size limit are queued (2 requests, 5 ms).
- Expired requests (deadline 0) return the fallback, and the model is never run.
- Batched scores equal unbatched scores (rtol 1e-5).
- The endpoint returns 200 for valid requests and 422 for empty or out-of-range candidate lists.

A cancellation bug was found during development: the batcher cancelled expired futures, so the awaiting
handler raised `CancelledError` and returned 500s under load. The fix is that the batcher skips expired
entries and `score()` handles its own timeout. A 3000-request burst with a 2 ms deadline now produces no exceptions.

## Caveats and substitutions

- **Ranker weights are random.** No trained model is available, so ranking quality is not measured. Only
  serving behaviour is.
- **Load generator on the same box.** The client, server, and torch threads share 2 cores. The reported
  ceilings (about 2,100–2,250 model rps) are for this setup and would be higher with separate hardware.
  The client is raw HTTP/1.1 over a 256-connection pool: an earlier httpx-based client saturated itself at
  about 100 rps, and those numbers were discarded.
- **Fallback = popularity prior.** The fallback is a Zipf-shaped score per item, not a learned popularity model.
- **CPU only.** The brief mentions GPU efficiency. Batching gains were measured on CPU only. On a GPU the
  per-call overhead, and so the gain from batching, would be larger.
- **No request-level metrics beyond `/stats`.** The batch histogram and fallback counts are cumulative.
