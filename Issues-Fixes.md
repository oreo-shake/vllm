# Issues and Fixes: prefix-pruning CPU offload lookup / recompute

## Context

GPU LRU + CPU `prefix_pruning_on_sessions` (or any CPU policy with
`supports_disjoint_spans=True`) stores KV on CPU in position tiers. Under
even modest CPU pressure, shallow tiers are peeled from **all** sessions
before deep tiers. That leaves a hole at the start of a prompt and one or
more later CPU runs (deferred spans).

LRU / ARC only do a contiguous prefix lookup, so they never take this path.
The span path is required for `prefix_pruning`, `prefix_pruning_on_sessions`,
and `first_request_pin`.

Example serve profile used while debugging:

```bash
VLLM_CPU_SESSION_TTL_K=2 VLLM_CPU_SESSION_TTL_S=120 VLLM_SERVER_DEV_MODE=1 \
vllm serve mistralai/Ministral-3-8B-Reasoning-2512 \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":53687091200,"block_size":256,"eviction_policy":"prefix_pruning_on_sessions","first_tier_tokens":16384,"tier_step_tokens":16384,"max_tiers":8}}'
```

Traffic: `scripts/prefix_cache_ttft_compare.py` (round 1 → wait →
`/reset_prefix_cache` → round 2). Reset clears GPU only unless
`reset_external=true`.

---

## Issue 1 — Async scheduling dropped deferred CPU spans (no load)

### Symptom

- Round 2 TTFT stayed cold (full recompute).
- Per-request log: `CPU=0`.
- Metrics: no `kv_offload_load_bytes`; only re-`store_bytes`.
- Same traffic with CPU `lru` showed large contiguous `CPU=` hits and loads.

### Root cause

Intended flow for a hole-at-front + later CPU run:

1. `_span_lookup` records deferred runs in `_offload_spans`.
2. `_clip_to_offload_span` recomputes only the hole.
3. `_requeue_for_offload_span` moves the request back to waiting when the
   cursor reaches the span.
4. `_match_offload_span` loads that CPU run.

Under **async scheduling**, `_update_after_schedule` advances
`num_computed_tokens` as soon as the last hole chunk is scheduled, while
`num_in_flight_tokens > 0`. The next overlapping schedule step then:

- Failed `_requeue_for_offload_span` (`num_in_flight_tokens != 0`).
- Called `_clip_to_offload_span(..., pause=False)` on the running path,
  which **popped the span and recomputed** the CPU-cached region.

So suffixes that were still on CPU were never loaded.

### Fix

In `vllm/v1/core/sched/scheduler.py`, the running-path clip uses
`pause=True`:

- While `num_computed_tokens` is already inside a deferred span but hole
  tokens are still in flight, clip returns `0` and **keeps** the span.
- After in-flight drains, `_requeue_for_offload_span` succeeds and the
  waiting path loads via `_match_offload_span`.

Waiting-path clip stays `pause=False` (stranded span → recompute), matching
the existing contract.

### Verification

- Unit tests: `tests/v1/core/test_offload_span_scheduling.py`.
- E2E: round-2 TTFT dropped ~78–79% vs round 1; `kv_offload_load_bytes`
  appeared in the server log.

---

## Issue 2 — Hit reporting ignored deferred spans

### Symptom

After Issue 1 was fixed, latency and `load_bytes` were healthy, but the
tier-hit log still showed `CPU=0`, and `External prefix cache hit rate`
looked far too low relative to bytes loaded.

### Root cause

The first-lookup path (and the `KV tier hits` log) only counted
**contiguous** external tokens from token 0. Deferred span lengths in
`_offload_spans` were omitted.

Aggregate external stats were also skewed: the first lookup recorded full
prompt queries with `hits=0`, then span loads recorded hits against a
smaller remaining query window.

### Fix

On first lookup in `scheduler.py`:

- `CPU` / external hits = contiguous external tokens + sum of pending
  span lengths (`_pending_offload_span_tokens`).
- Prefill `num_external_cached_tokens` uses the same total.
- Log format:
  `GPU=%d CPU=%d (contiguous=%d deferred=%d)`.

On the later span-load path, connector prefix-cache queries/hits are not
recorded again (already counted at discovery), so the external hit rate
is not double-counted or diluted.

### Expected log after fix

```text
Request ... KV tier hits: GPU=0 CPU=192512 (contiguous=0 deferred=192512)
```

Contiguous may still be `0` when prefixes were peeled; that is expected for
this eviction policy. `CPU` should reflect the deferred runs that will load
after the hole is recomputed.
