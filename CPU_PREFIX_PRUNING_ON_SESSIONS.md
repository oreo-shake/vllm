# CPU tier: `prefix_pruning_on_sessions`

This document explains the **CPU offload** eviction policy
`prefix_pruning_on_sessions` — how it decides what to keep when many
conversations (`session_id`s) are active and sending requests, and why the
code is structured that way.

Primary implementation:

- `vllm/v1/kv_offload/cpu/policies/prefix.py`
  (`PrefixPruningOnSessionsCachePolicy`, base `PrefixPruningCachePolicy`)
- `vllm/v1/core/session_liveness.py` (`SessionLivenessTracker`)
- `vllm/v1/kv_offload/cpu/manager.py` (`CPUOffloadingManager`)
- Connector touch / span lookup:
  `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`

Related overview (GPU + CPU): [`TIER_EVICTION_ALGORITHM.md`](TIER_EVICTION_ALGORITHM.md).

---

## 1. Intent

The CPU tier is a large but finite host-memory cache of KV chunks that left
the GPU. Under ordinary LRU, the **oldest** free chunk is the next victim.
For long agent / chat turns that share a growing suffix, that is the wrong
loss order: the deep end of a prompt is often what the next turn will reuse,
while the shallow prefix can be recomputed or is less valuable once capacity
is tight.

`prefix_pruning_on_sessions` changes two things:

1. **Prefix peel (position tiers).** Among live capacity, evict shallow
   token positions before deep ones, so a reused **suffix** tends to survive
   while heads of many sequences are sacrificed first.
2. **Dead-session preference.** When a conversation has been idle long
   enough that the CPU session TTL declares it dead, **all** of that
   session’s remaining unique CPU chunks (suffix included) become preferred
   victims before live sessions’ later tiers. Idle conversations stop
   hoarding a protected deep suffix.

The policy does **not** change how much CPU memory exists. It only changes
**who loses first** when a new store needs slots.

A follow-up turn that continues from the end of a previous context can still
hit a resident suffix even after the prefix was peeled — the connector loads
that later run as a **disjoint span** (see §8).

---

## 2. Enabling the policy

Select the policy in `kv_connector_extra_config`:

```bash
vllm serve "$MODEL" \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 53687091200,
      "block_size": 256,
      "eviction_policy": "prefix_pruning_on_sessions",
      "first_tier_tokens": 16384,
      "tier_step_tokens": 16384,
      "max_tiers": 8
    }
  }'
```

| Knob | Default | Role |
| --- | --- | --- |
| `eviction_policy` | `lru` | Must be `prefix_pruning_on_sessions`. |
| `first_tier_tokens` | `16384` | Width of tier 0 (leading tokens). |
| `tier_step_tokens` | `16384` | Width of each later position tier. |
| `max_tiers` | `8` | Position tiers before the CPU LRU tail. |
| `VLLM_CPU_SESSION_TTL_S` | unset (`0`) | Floor for “session idle → prefer full wipe”. |
| `VLLM_CPU_SESSION_TTL_K` | `2` | Multiplier on that session’s idle-gap EWMA. |

Factory registration: `CachePolicyFactory` name
`prefix_pruning_on_sessions` → `PrefixPruningOnSessionsCachePolicy`
(`vllm/v1/kv_offload/cpu/policies/factory.py`).

Clients should send a stable `session_id` (body field, `X-Session-ID`, or
`extra_args["session_id"]`). Without an id, chunks still participate in
**position-tier peel**, but they never enter the dead-session set.

---

## 3. What the policy remembers

Per CPU chunk key (`OffloadKey`):

| State | Meaning |
| --- | --- |
| `chunks[key]` | Physical slot (`ChunkStatus`: `ref_cnt`, `chunk_id`). |
| `_positions[key]` | Start-token index used for tiering (from connector touch). |
| `_ranks[key]` | Recency rank (larger = newer). Used inside a tier and in the LRU tail. |
| `_sessions[key]` | Last `session_id` that touched / stored this key. |
| `_session_keys[sid]` | Inverse index: keys currently attributed to that session. |
| `_evictable` | Keys with `ref_cnt == 0` (not pinned by an in-flight load/store). |

`PrefixPruningOnSessionsCachePolicy` also owns a **CPU-local**
`SessionLivenessTracker(observe_idle=True)`. Its floor is
`VLLM_CPU_SESSION_TTL_S` (or an explicit `session_ttl_s` in tests). This is
**not** the GPU dead-session detector; a session the GPU already called dead
can still be live on CPU until the CPU TTL expires.

Optional: `attach_session_tracker(gpu_tracker)` lets the policy read the
GPU-side idle-gap record instead of (or as) its local tracker, so GPU LRU
can still feed idle averages into CPU session scoring. When a shared
tracker is attached, CPU `on_new_request` / `notify_request_finished` do
**not** double-count gaps on the local tracker.

---

## 4. Position tiers (the peel)

`_TierScorer` maps a start-token `position` into a tier:

- `position ≤ first_tier_tokens` → tier `0`
- else → `1 + (position - first_tier_tokens - 1) // tier_step_tokens`
- unknown position, or tier ≥ `max_tiers` → **LRU tail** (recency only)

**Intent:** peel earliest remaining prefixes first. Inside a position tier,
earlier token positions go before later ones; ties break on LRU rank
(smaller rank = older = preferred victim). Past `max_tiers`, deep body
blocks behave like CPU LRU so unbounded depth does not freeze the cache.

Positions come from connector `_touch`: for each chunk index `i`,
`position = i * tokens_per_chunk` (start token). The policy deliberately
does **not** fall back to end-token `get_offload_key_position`, which would
push a chunk at a tier boundary into a deeper tier and change peel order.

Recency refresh: `refreshes_recency_from_connector = True`. Mid-request
connector touch owns LRU order (same contract as plain `prefix_pruning`).
`mark_evictable` when a store finishes does **not** bump rank, so a late
store completion cannot make a chunk look newer than one inserted earlier.

---

## 5. Session identity across concurrent requests

### 5.1 Binding a chunk to a session

On `touch` and `note_stored`, the policy reads
`ReqContext.session_id` (or `kv_transfer_params["session_id"]`) and calls
`_note_session(key, session_id)`:

- If the key moves from session A to B, it is removed from A’s set and
  added to B’s.
- Keys with no session id are left unbound; they never score as
  “dead session.”

**Intent under multi-session load:** attribution is **per chunk key**, not
per request isolation. The CPU cache is global. Many sessions may share
hash-identical prefix chunks (common system prompt). The last session that
touched a shared key owns the session tag for eviction scoring. That is
acceptable because:

- Shared stems are usually shallow → they sit in early tiers anyway.
- Dead-session preference is a **hint to free capacity from abandoned
  conversations**, not a hard partition of the cache into per-session
  quotas.

### 5.2 Liveness updates (arrivals and finishes)

For each request with a `session_id`:

1. Manager `on_new_request` → policy `on_new_request` →
   `tracker.on_arrival(session_id, now, priority=…)`.
2. While the request runs, connector `touch` refreshes positions, session
   binding, and LRU ranks for keys that request sees.
3. On finish, `notify_request_finished` → `tracker.on_finish(session_id, now)`.

The idle gap sample is **completion → next arrival of the same session**,
not request duration. Concurrent in-flight turns of the same session do not
count a gap while `last_arrival ≥ last_finish` (a turn is still “in flight”).

**Multiple sessions in flight at once:** each session has its own row in
`tracker.states`. Session A finishing does not age session B. Eviction
compares candidates from **all** sessions’ unbound/`ref_cnt==0` chunks in
one global heap (§6).

### 5.3 Dead-session predicate

```text
idle_limit = max(VLLM_CPU_SESSION_TTL_S, VLLM_CPU_SESSION_TTL_K * ewa_gap)
            # if no gaps yet: floor only (or “not expired” if floor is 0)
dead ⇔  (now - last_finish) ≥ idle_limit
        and no later arrival after that finish
```

Until a session has at least one observed gap, a positive
`VLLM_CPU_SESSION_TTL_S` is the only limit. With neither floor nor gap,
`idle_expired` is false — the session is never wiped by the session rule
(only ordinary prefix peel / LRU tail apply).

---

## 6. Eviction scoring when many sessions compete

`evict(n, protected)` builds one candidate per key in
`_evictable - protected`, scores it, and takes the `n` smallest scores
(`heapq.nsmallest`). If fewer than `n` candidates exist, eviction fails
atomically (`None`) and the store is rejected — nothing is half-evicted.

Score tuple from `_TierScorer.score(position, rank, key, is_dead)`:

| Priority group | Tuple shape | Meaning |
| --- | --- | --- |
| Dead session | `(0, position, rank, 0, key)` | Wipe first, even deep suffixes. Inside the dead set, earlier positions still go first. |
| Position tier | `(1, tier, position, rank, key)` | Live (or unbound) chunks: shallow tier → earlier position → older rank. |
| LRU tail | `(2, 0, rank, 0, key)` | Past `max_tiers` / unknown position: pure recency. |

**Multi-session intent, in order:**

1. **Reclaim abandoned conversations** before peeling more of a live one.
   Session D finished a long time ago; sessions L1…Ln keep sending turns.
   When L1 needs to store new chunks, D’s remaining CPU blocks (including
   deep unique suffixes) are preferred victims over L1’s shallow tier-0
   blocks — and especially over L1’s deep suffix.
2. **Among live sessions**, do not pick “oldest session globally.” Pick
   **shallowest remaining prefix** across everyone. So under concurrency,
   many live sessions lose heads together while each keeps its deep end
   as long as capacity allows.
3. **Inside one session**, peel front → back (tier, then position), same
   as plain `prefix_pruning`.
4. **Protected keys** (the store’s own offer, or `ref_cnt > 0` pins from
   `prepare_load` / `pin_deferred`) are never candidates. Concurrent loads
   and admitted deferred spans cannot be stolen by another session’s store.

If eviction cannot free `n` slots, `prepare_store` returns `None` and that
request’s offload store is skipped for this attempt — better than corrupting
another session’s in-flight transfer.

---

## 7. Lifecycle under concurrent multi-session traffic

```text
Client A (session=sA)     Client B (session=sB)     CPU policy / manager
        |                         |                         |
        |-- new request --------->|                         |
        |                         |-- new request --------->|
        |                         |         on_arrival(sA), on_arrival(sB)
        |                         |                         |
        |-- lookup / touch -------|------------------------>|
        |                         |-- lookup / touch ------>|
        |                         |    bind keys→sA/sB, positions, bump ranks
        |                         |                         |
        |-- prepare_store ------->|                         |
        |                         |    may evict: dead sessions first,
        |                         |    then shallow tiers of anyone live
        |                         |    note_stored(keys, sA)
        |                         |-- prepare_store ------->|
        |                         |    same global heap; protect B's offer
        |                         |                         |
        |-- finish -------------->|                         |
        |                         |         on_finish(sA) → last_finish
        |                         |-- finish -------------->|
        |                         |         on_finish(sB)
        |                         |                         |
        |  (idle > TTL)           |  (still turning)        |
        |                         |-- prepare_store ------->|
        |                         |    sA chunks score as dead (group 0)
```

Important properties:

- **No per-session capacity quota.** Fairness is only through the score
  order above. A hot session that keeps touching its keys refreshes ranks
  and keeps resetting “in flight” so it does not look dead.
- **Stores compete for one free list.** Whichever request’s
  `prepare_store` runs first in the scheduler step sees the current
  evictable set; the other sees whatever remains after that eviction.
- **Loads pin victims out of the set** (`ref_cnt` / `mark_non_evictable`).
  A session mid-H2D load cannot lose those chunks to another session’s
  store. Deferred span loads after admit use `pin_deferred` for the same
  reason (hole-store must not chase a moving suffix — see
  `Issues-Fixes.md` Issue 5).
- **Shared prefix hashes** are single physical chunks. Evicting them once
  removes them for every session that would have hit that hash. The peel
  still prefers shallow tiers, which is usually where shared stems live.

---

## 8. Why disjoint spans matter

`supports_disjoint_spans = True`. After prefix peel, a sequence may look
like:

```text
[ missing head | hole | cached deep run | … ]
```

LRU/ARC lookup stops at the first miss (`_maximal_prefix_lookup`). Prefix
policies instead return later cached runs via `cached_spans` /
`_span_lookup`. The scheduler:

1. Reports contiguous external hits from the GPU cursor (may be 0).
2. Records later runs as deferred spans.
3. Computes (or recomputes) the hole, then requeues so the connector loads
   the next resident run as a normal contiguous hit.

**Intent:** the eviction policy is allowed to punch holes in the prefix
without making the surviving suffix unreachable. Without span support,
peeling the head would look like a total miss and force full recompute —
defeating the whole point of keeping the deep end.

---

## 9. Comparison to sibling CPU policies

| Policy | Session wipe | First-request hold | Peel |
| --- | --- | --- | --- |
| `prefix_pruning` | No (tracker unused unless shared) | No | Position tiers → LRU tail |
| `prefix_pruning_on_sessions` | Yes (CPU TTL / idle EWMA) | No | Same peel; dead sessions score first |
| `first_request_pin` | Uses same idle notion for pin expiry | Priority-0 first request hashes held until idle | Same peel for everything else |
| `lru` / `arc` | No | No | Recency / ARC; contiguous lookup only |

`prefix_pruning_on_sessions` is the multi-tenant default when you want
suffix preservation **and** automatic reclaim of conversations that stopped
sending requests, without permanently pinning a stem.

---

## 10. Worked example (two live sessions + one dead)

Assume `first_tier_tokens = tier_step_tokens = 1`, `max_tiers = 4`,
`session_ttl_s = 5`, and three keys:

| Key | Session | Position | Notes |
| --- | --- | --- | --- |
| `dead_shallow` | `dead` | 0 | Finished; idle ≫ TTL |
| `live_shallow` | `live` | 0 | Still turning |
| `live_deep` | `live` | 100 | Still turning |

Need 1 free slot for a new store from `live`:

1. `dead_shallow` scores group `0` (dead) → **evicted**.
2. If more space is needed, `live_shallow` (tier 0) before `live_deep`.
3. `live_deep` survives so the next turn of `live` can still span-load the
   suffix after recomputing the peeled head.

If both sessions are live and neither is dead, eviction is pure prefix peel
across both: shallow keys of either session before deep keys of either
session, with LRU rank breaking ties.

---

## 11. Operational checklist

1. Set `eviction_policy` to `prefix_pruning_on_sessions` and size
   `cpu_bytes_to_use` for the working set of **live** suffixes you care
   about (dead sessions will be preferred victims once idle).
2. Set `VLLM_CPU_SESSION_TTL_S` (and optionally `VLLM_CPU_SESSION_TTL_K`) so
   “user left” matches your think-time distribution; leave unset if you
   only want peel without session wipe.
3. Clients must send a stable `session_id` per conversation for the wipe
   path to work.
4. Expect `contiguous=0` with large `deferred=` hit counts when prefixes
   were peeled — that is normal for this policy.
5. Pair with GPU suffix eviction independently; GPU and CPU TTLs are
   separate (`VLLM_SESSION_*` vs `VLLM_CPU_SESSION_*`).

---

## 12. Code map (intent → symbol)

| Intent | Where |
| --- | --- |
| Choose policy | `CachePolicyFactory` / `CPUOffloadingSpec.get_manager` |
| Tier math | `_TierScorer.score` |
| Bind session + position | `touch`, `note_stored`, `_note_session`, `_note_position` |
| Idle gaps | `SessionLivenessTracker.on_arrival` / `on_finish` / `idle_expired` |
| Prefer dead then peel | `PrefixPruningCachePolicy.evict` + `_dead_sessions` |
| CPU-local tracker + floor | `PrefixPruningOnSessionsCachePolicy.__init__` |
| Fail store if cannot free | `CPUOffloadingManager.prepare_store` |
| Pin deferred suffix | `pin_deferred` / connector admit path |
| Load hole then suffix | `supports_disjoint_spans`, scheduler `_span_lookup` |
| Unit behavior | `tests/v1/kv_offload/cpu/test_cpu_tier_policies.py` |
