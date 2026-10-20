# Tier eviction algorithm

Branch `feat/idle-demote-protect-suffix` in `opensource/eviction/vllm`.

Native vLLM keeps free GPU KV blocks in one doubly linked list. Allocation always takes the head. A freed cached block is appended at the tail, so the head is the least recently freed cached block. That is LRU: the next victim is whatever has been unused the longest, with no notion of where the block sat inside a prompt.

This branch keeps that tail and puts one or more **evict-first tiers** in front of it (`TieredFreeKVCacheBlockQueue`). A block’s tier is chosen from its token position when the request is freed. The three policies below are three different rules for which blocks enter which tier, and for when a block is later moved. LRU is still the last tier.

Let a freed request have blocks `b0, b1, ..., b(n-1)` in token order. Block size is `B` tokens. `T0` is `VLLM_SUFFIX_EVICTION_TIER_TOKENS` (leading tokens that form tier 0). `S` is `VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS` (tokens per later tier; defaults to `T0`). `M` is `VLLM_SUFFIX_EVICTION_MAX_TIERS` (0 means uncapped).

$$
L_0 = \lfloor T_0 / B \rfloor, \quad L = \lfloor S / B \rfloor
$$

$$
tier(i) =
\begin{cases}
0 & i < L_0 \\
\min\left(1 + \lfloor (i - L_0) / L \rfloor, N\right) & L > 0 \\
N & L = 0
\end{cases}
$$

`N` is the index of the LRU tail. Eviction order is total: every block of tier `k` is taken before any block of tier `k+1`. Inside a tier, earlier positions are freed first, so a single request is peeled from the front of the prompt toward the end.

Compared with LRU, the victim is no longer “oldest free block.” It is “shallowest remaining prefix,” and only after those prefixes are gone does recency decide. The useful consequence is that the **tail of a long request stays resident** while the heads of many requests are dropped. A follow-up turn that continues from the end of the previous context can still hit that suffix. LRU would have dropped that suffix as soon as it was the oldest free span, even if the prefix was the part nobody will reuse.

## 1. Prefix pruning

Prefix pruning is the position rule above, with no session identity.

On free, leading blocks join tier 0 and each following span of `L` blocks joins the next tier. Blocks past the last configured tier join the LRU tail, so a very deep body is still recency-governed. If `TIER_TOKENS` is 0, the older fraction mode is used: the first `VLLM_SUFFIX_EVICTION_PREFIX_FRACTION` of the request is tier 0 and the rest is the LRU tail.

Idle aging (`VLLM_SUFFIX_EVICTION_IDLE_TTL_S`, mode `VLLM_SUFFIX_EVICTION_IDLE_DEMOTE`) then moves blocks that have sat free longer than the TTL:

- `all`: every expired block jumps to tier 0, suffix included.
- `protect`: the same jump, except the last `VLLM_SUFFIX_EVICTION_PROTECT_TOKENS` of that request (`idle_protect`) stay where they are.
- `step`: each expired unprotected block moves one tier toward 0, `tier = max(0, tier - 1)`, once per TTL. A suffix that started in a high tier needs many idle intervals before it becomes evict-first.

What this changes versus LRU is the **order of loss**, not the amount of cache. Under a fixed GPU budget the prefix of every finished request is the first thing sacrificed, so the same number of blocks buys more suffix hits and fewer full recomputes of a long tail. The cost is that a popular short prefix (a shared system prompt that is *not* the suffix) can be peeled before a stale suffix, which is why the later two policies exist.

## 2. Prefix pruning with a session tier

Same position tiers, plus a per-`session_id` liveness test. A session is the conversation that reuses one id across turns.

Two detectors, both floored by `VLLM_SUFFIX_SESSION_DEAD_TTL_S` (`T_min`):

**Per-session EWA** (`VLLM_SUFFIX_SESSION_DETECTOR=ewa`). Gaps are completion time to the next arrival. With smoothing `alpha` (`VLLM_SUFFIX_SESSION_EWA_ALPHA`) and `k` (`VLLM_SUFFIX_SESSION_DEAD_K`):

$$
g_t = \alpha \Delta_t + (1-\alpha) g_{t-1}, \quad q_t = \alpha \Delta_t^2 + (1-\alpha) q_{t-1}
$$

$$
\sigma_t = \sqrt{\max(0, q_t - g_t^2)}
$$

$$
dead \iff idle \ge \max(T_{min}, g_t + k \sigma_t)
$$

until `VLLM_SUFFIX_SESSION_MIN_GAPS` samples exist, the threshold is just `T_min`.

**Server-wide quantile** (`global`). One reservoir of the last `VLLM_SUFFIX_SESSION_GAP_WINDOW` gaps. The threshold is `max(T_min, Q_p)` at `VLLM_SUFFIX_SESSION_QUANTILE`. One session’s short thinks cannot lower its own bar.

When the test says dead, **every** remaining unique block of that session is prepended to tier 0, suffix included (`promote_dead_session_blocks`). That is a full wipe, not another prefix peel. Live sessions stay on their position tiers.

Versus plain prefix pruning, a session that has gone quiet no longer occupies a high tier because its suffix is deep. Its whole footprint becomes the next victim, so capacity returns to sessions that are still turning. Versus LRU, a live session still keeps its suffix while only its prefix is peeled; a dead session does not get that privilege. This detector does not record or defer the first request. That stays off while `VLLM_SESSION_TTL_LIMIT_S` is 0.

## 3. First request pinned

The third rule is which request, inside a session, is held back from that peel. It does **not** turn on with prefix pruning or with the session-dead detector. It turns on only when `VLLM_SESSION_TTL_LIMIT_S` is greater than 0 (suffix eviction itself must also be on, because that is what installs the tiered queue).

`priority` defaults to **0** when the request body does not contain the field (`Request` coerces `None` to 0; the OpenAI protocol field default is already 0). That default only matters for this policy. For a priority-0 session the block hashes of the **first** request are recorded at arrival, or sealed on the first completion if the arrival snapshot was empty. Later turns do not replace that snapshot. Priority 1 and above record nothing.

On free, a block whose hash is in that set is marked `session_protected`. The tiered free queue still *contains* it (no extra reference, so it cannot leak a block forever), but `popleft` takes every unprotected free block first and only then those protected blocks. GPU pressure can still reclaim the first request; it is the last resort, not a hard pin.

Protection ends when the session is predicted dead or marked stale (the priority-0 TTL cap, or the think-time detector). A later arrival revives the same first-request hashes. Session-dead promotion clears the flag and prepends the blocks to tier 0.

What this adds on top of prefix pruning: the first turn is usually the shared stem (system prompt, tool spec, the part every later turn extends). Position tiers would peel that stem first, because it is the prefix. Deferring it means later turns keep hitting the stem while *other* sessions’ shallow prefixes are the ones that go. The advantage over LRU is the same suffix bias as policy 1, plus the stem of a priority-0 session is not the first victim of its own prefix peel. The advantage over a hard pin (`VLLM_PIN_SYSTEM_PROMPT`, which holds an extra ref and is off by default) is that a full GPU still makes progress: protected blocks are only a preference, and a dead session drops them.

## How the three sit on the LRU list

```
never-allocated  →  tier 0 (shallow prefixes, dead sessions)  →  tier 1  →  …  →  LRU tail
```

The priority-0 first request is deferred to the end of that order only while policy 3 is configured. Policies 1 and 2 leave `session_protected` unset, so those blocks are ordinary tier members.

| Policy | What moves relative to LRU | What you gain |
| --- | --- | --- |
| Prefix pruning | Victim = smallest token index, not oldest free time | Suffix of a long request survives while heads are dropped |
| Session tier | Dead session jumps entirely to tier 0 | Idle conversations stop hoarding a protected suffix |
| First-request pin | Priority 0 (the omitted-field default) keeps request 1 until nothing else is free | The stem later turns extend stays hit-able without a permanent ref |

## Knobs

All three policies require prefix caching, one full-attention KV group, and `VLLM_SUFFIX_PRESERVING_EVICTION=1`. Leaving that at 0 keeps native LRU. The other two policies are extra flags on top of that switch. Setting only the session-dead knobs, or only the tier knobs, does not pin the first request.

### Policy 1 — prefix pruning

| Knob | Default | Role |
| --- | --- | --- |
| `VLLM_SUFFIX_PRESERVING_EVICTION` | `0` | Must be `1`. |
| `VLLM_SUFFIX_EVICTION_TIER_TOKENS` | `0` | Leading tokens placed in tier 0. `0` falls back to the fraction below. |
| `VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS` | `0` | Tokens per tier after tier 0. `0` reuses `TIER_TOKENS`. |
| `VLLM_SUFFIX_EVICTION_MAX_TIERS` | `0` | Cap on position tiers. `0` means one tier per step across the model length. Past the cap, blocks join the LRU tail. |
| `VLLM_SUFFIX_EVICTION_PREFIX_FRACTION` | `0.5` | Used only when `TIER_TOKENS` is 0. Fraction of the request, from the first block, that is tier 0. |
| `VLLM_SUFFIX_EVICTION_IDLE_TTL_S` | `0` | Seconds a free block may sit before idle demotion. `0` disables demotion. |
| `VLLM_SUFFIX_EVICTION_IDLE_DEMOTE` | `all` | `all`, `protect`, or `step`. Ignored when the TTL is 0. |
| `VLLM_SUFFIX_EVICTION_PROTECT_TOKENS` | `0` | Tail tokens skipped by `protect` and `step`. `0` disables that skip. |

Example, tiers only, no session logic and no first-request pin:

```bash
export VLLM_SUFFIX_PRESERVING_EVICTION=1
export VLLM_SUFFIX_EVICTION_TIER_TOKENS=16384
export VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS=16384
export VLLM_SUFFIX_EVICTION_MAX_TIERS=8
```

### Policy 2 — prefix pruning with a session tier

Everything in policy 1, plus a dead-session floor above 0. `VLLM_SESSION_TTL_LIMIT_S` stays 0, so the first request is not pinned.

| Knob | Default | Role |
| --- | --- | --- |
| `VLLM_SUFFIX_SESSION_DEAD_TTL_S` | `0` | Must be `> 0` to enable the detector. Floor on the dead threshold. |
| `VLLM_SUFFIX_SESSION_DETECTOR` | `ewa` | `ewa` (per session) or `global` (server-wide quantile). |
| `VLLM_SUFFIX_SESSION_EWA_ALPHA` | `0.3` | Gap smoothing for `ewa`. |
| `VLLM_SUFFIX_SESSION_DEAD_K` | `3` | Stddev multiplier for `ewa`. |
| `VLLM_SUFFIX_SESSION_MIN_GAPS` | `2` | Gaps required before the adaptive threshold replaces the floor. |
| `VLLM_SUFFIX_SESSION_QUANTILE` | `0.95` | Quantile for `global`. |
| `VLLM_SUFFIX_SESSION_GAP_WINDOW` | `256` | How many recent gaps `global` keeps. |

Requests need a `session_id` (body field, `X-Session-ID`, or `extra_args["session_id"]`). Without an id there is nothing to mark dead.

Example:

```bash
export VLLM_SUFFIX_PRESERVING_EVICTION=1
export VLLM_SUFFIX_EVICTION_TIER_TOKENS=16384
export VLLM_SUFFIX_SESSION_DEAD_TTL_S=30
export VLLM_SUFFIX_SESSION_DETECTOR=ewa
```

### Policy 3 — first request pinned

Policy 1 must be on, and `VLLM_SESSION_TTL_LIMIT_S` must be `> 0`. That single inequality is what sets `pin_first_request`. Policy 2’s dead-TTL may be 0 or set; it does not by itself pin anything.

| Knob | Default | Role |
| --- | --- | --- |
| `VLLM_SESSION_TTL_LIMIT_S` | `0` | Must be `> 0`. Enables the pin. GPU session TTL is `max(T_limit, K * g)`. |
| `VLLM_SESSION_TTL_K` | `1` | Multiplier on the priority-0 gap EWMA. Uses `VLLM_SUFFIX_SESSION_EWA_ALPHA`. |

A body with no `priority` field is priority 0, so the first request of that session is the one deferred. Send `priority >= 1` to opt that session out. The pin lasts until the priority-0 cap (or the policy-2 detector, if that floor is also set) marks the session dead.

Example, pin without the think-time wipe:

```bash
export VLLM_SUFFIX_PRESERVING_EVICTION=1
export VLLM_SUFFIX_EVICTION_TIER_TOKENS=16384
export VLLM_SESSION_TTL_LIMIT_S=60
export VLLM_SESSION_TTL_K=1
```

`VLLM_PIN_SYSTEM_PROMPT` is a different mechanism (extra reference on a detected shared prefix). The scheduler does not enable it for these three policies.

## CPU tier

The built-in CPU offload manager selects a `CachePolicy` with `kv_connector_extra_config["eviction_policy"]`. `lru` and `arc` are unchanged: they do not record sessions, they do not defer a first request, and lookup stays one contiguous run from the GPU cursor. GPU LRU (`VLLM_SUFFIX_PRESERVING_EVICTION=0`) does not read these CPU settings. Any GPU mode can pair with any CPU mode.

The three extra names are `prefix_pruning`, `prefix_pruning_on_sessions`, and `first_request_pin`.

Position tiers use the same rule as the GPU display math above. Token index comes from the offload chunk. Shallow tiers are evicted before the CPU LRU tail. `prefix_pruning` ignores `session_id`.

`prefix_pruning_on_sessions` adds a CPU-local liveness tracker. It is not the GPU tracker. A session the GPU has already called dead can still be live on CPU until the CPU TTL. Dead CPU sessions are evicted before live later tiers.

`first_request_pin` keeps LRU order and, for priority 0, refuses to evict the first request's hashes until the CPU session TTL. A missing `priority` field is 0. Priority 1 and above are ordinary LRU victims. The hashes are recorded when the session arrives. They become protected only once those blocks are stored on CPU, which is the copy that left the GPU. While the TTL holds, a store that would have to evict them fails instead of dropping them. After the TTL, they are ordinary LRU victims.

The idle gap is the time from one request of the session finishing to the next request of that same session arriving. `g` is the exponential average of those successive gaps, not an average of request durations. Smoothing `alpha` is `VLLM_SUFFIX_SESSION_EWA_ALPHA`:

$$
g_t = \alpha \Delta_t + (1-\alpha) g_{t-1}
$$

On both tiers the session TTL used by the session policy and the first-request pin is the larger of that average and the tier bound, so an outlier gap is honored:

$$
T = \max(K \cdot g_t, T_{tier})
$$

`K` is `VLLM_SESSION_TTL_K` on GPU and `VLLM_CPU_SESSION_TTL_K` on CPU (default 2). `T_tier` is `VLLM_SESSION_TTL_LIMIT_S` on GPU and `VLLM_CPU_SESSION_TTL_S` on CPU. `K` multiplies the moving average only. It does not scale the other tier's TTL. Until the session has one gap, `T` is the tier bound. With no bound and no gap, the session is not declared idle.

Current idle time is `now - last completion`, and only while no later request has already arrived. That clock is the same on GPU and CPU. A session is eligible for that tier's session eviction when current idle time is at least `T`. Otherwise its blocks stay protected by that rule. GPU LRU does not consult `T`. The GPU dead-session detector, when its own floor is set, still uses `max(T_min, g + k sigma)` and is a separate switch from this bound.

That average is recorded on the GPU block pool whenever either tier is not LRU. Prefix pruning on GPU records it even if the dead detector is off. GPU LRU plus a non-LRU CPU policy still records it, and the CPU policy reads that same tracker. The GPU free queue stays the LRU list, so GPU LRU eviction does not change. CPU LRU and ARC do not read the tracker, and the tracker is not created when both tiers are LRU or ARC.

### Lookup and recompute

`lru` and `arc` keep `_maximal_prefix_lookup`: one contiguous hit count, then stop.

The three new policies set `supports_disjoint_spans`. Lookup still loads the contiguous run that starts at the GPU cursor through the existing external-token path. Later cached runs are returned separately. The scheduler computes only up to the next run. When that cursor is reached and nothing is in flight, the request goes back to the waiting queue so the connector loads that run as a normal contiguous hit, then the next hole is computed. GPU suffix hits keep their own jump, because those tokens are already on GPU.

### CPU knobs

| Knob | Default | Role |
| --- | --- | --- |
| `eviction_policy` | `lru` | `lru`, `arc`, `prefix_pruning`, `prefix_pruning_on_sessions`, or `first_request_pin`. |
| `first_tier_tokens` | `16384` | Tier 0 width for the two prefix policies. |
| `tier_step_tokens` | `16384` | Width of each later prefix tier. |
| `max_tiers` | `8` | Position tiers before the CPU LRU tail. |
| `VLLM_CPU_SESSION_TTL_S` | unset | CPU tier bound inside `max(K * idle EWMA, bound)`. |
| `VLLM_CPU_SESSION_TTL_K` | `2` | Factor on the session idle-gap average. Not a scale of the GPU TTL. |

## Serve a profile

`benchmark/serve_cross_tier.sh` is the same launch as the commands below. GPU policy is `lru`, `prefix`, `session`, or `pin`. CPU policy is `lru`, `arc`, `prefix`, `session`, or `pin`. CPU offload is always attached (8GiB unless `CPU_BYTES` is set).

```bash
./benchmark/serve_cross_tier.sh lru pin 0 8000
./benchmark/serve_cross_tier.sh prefix session 0 8001
```

Shared `vllm serve` flags for every profile, from `opensource/eviction/vllm`:

```bash
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-model-len 262144 \
  --tensor-parallel-size 1
```

Add the env block and, when a CPU tier is in the pair, one `--kv-transfer-config`. `CPU_BYTES` is `8589934592`.

### GPU LRU, CPU first-request pin

```bash
export VLLM_SUFFIX_PRESERVING_EVICTION=0
export VLLM_CPU_SESSION_TTL_S=120
export VLLM_CPU_SESSION_TTL_K=2
vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-model-len 262144 --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592,"eviction_policy":"first_request_pin","first_tier_tokens":16384,"tier_step_tokens":16384,"max_tiers":8}}'
```

### GPU prefix pruning, CPU session tiers

```bash
export VLLM_SUFFIX_PRESERVING_EVICTION=1
export VLLM_SUFFIX_EVICTION_TIER_TOKENS=16384
export VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS=16384
export VLLM_SUFFIX_EVICTION_MAX_TIERS=16
export VLLM_SUFFIX_EVICTION_IDLE_TTL_S=60
export VLLM_CPU_SESSION_TTL_S=120
export VLLM_CPU_SESSION_TTL_K=2
vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8001 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-model-len 262144 --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592,"eviction_policy":"prefix_pruning_on_sessions","first_tier_tokens":16384,"tier_step_tokens":16384,"max_tiers":8}}'
```

### Other pairs

Compose the GPU env from the matching block and set `eviction_policy` in the same JSON. Omit `--kv-transfer-config` only when you want the GPU-only server from `benchmark/serve.sh`.

| Profile | Extra env on top of the GPU tiers above | `eviction_policy` |
| --- | --- | --- |
| GPU `lru`, CPU `lru` | `VLLM_SUFFIX_PRESERVING_EVICTION=0` only | `lru` |
| GPU `lru`, CPU `prefix` | `VLLM_SUFFIX_PRESERVING_EVICTION=0` | `prefix_pruning` |
| GPU `prefix`, CPU `lru` | the prefix env block | `lru` |
| GPU `session`, CPU `lru` | prefix env, plus `VLLM_SUFFIX_SESSION_DEAD_TTL_S=30`, `VLLM_SUFFIX_SESSION_DETECTOR=ewa`, `VLLM_SUFFIX_SESSION_EWA_ALPHA=0.3`, `VLLM_SUFFIX_SESSION_DEAD_K=3`, `VLLM_SUFFIX_SESSION_MIN_GAPS=2` | `lru` |
| GPU `pin`, CPU `lru` | prefix env, plus `VLLM_SESSION_TTL_LIMIT_S=60`, `VLLM_SESSION_TTL_K=2`, `VLLM_SUFFIX_SESSION_EWA_ALPHA=0.3` | `lru` |
| GPU `session`, CPU `pin` | the session env, plus `VLLM_CPU_SESSION_TTL_S=120` and `VLLM_CPU_SESSION_TTL_K=2` | `first_request_pin` |
| GPU `pin`, CPU `session` | the pin env, plus `VLLM_CPU_SESSION_TTL_S=120` and `VLLM_CPU_SESSION_TTL_K=2` | `prefix_pruning_on_sessions` |

`arc` uses the same command as CPU `lru` with `"eviction_policy":"arc"`. CPU `session` and CPU `pin` need the CPU TTL env. CPU `lru`, `arc`, and `prefix` do not.

