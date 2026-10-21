# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.policies.prefix import (
    FirstRequestPinCachePolicy,
    PrefixPruningCachePolicy,
    PrefixPruningOnSessionsCachePolicy,
)


def _key(name: bytes) -> bytes:
    return make_offload_key(name, 0)


def test_lru_has_no_spans_and_ignores_session():
    policy = LRUCachePolicy(4)
    assert not policy.supports_disjoint_spans
    assert policy.cached_spans([_key(b"a")]) is None


def test_prefix_pruning_evicts_shallow_before_deep():
    policy = PrefixPruningCachePolicy(
        4, first_tier_tokens=1, tier_step_tokens=1, max_tiers=4
    )
    shallow = _key(b"s")
    deep = _key(b"d")
    shallow_block = ChunkStatus(0)
    deep_block = ChunkStatus(1)
    shallow_block.ref_cnt = 0
    deep_block.ref_cnt = 0
    policy.insert(shallow, shallow_block)
    policy.insert(deep, deep_block)
    ctx = ReqContext(
        req_id="r",
        token_positions={shallow: 0, deep: 100},
    )
    policy.touch([shallow, deep], ctx)
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == shallow
    assert policy.get(deep) is not None


def test_session_policy_prefers_dead_session():
    policy = PrefixPruningOnSessionsCachePolicy(
        4,
        first_tier_tokens=10_000,
        tier_step_tokens=10_000,
        max_tiers=4,
        session_ttl_s=5,
    )
    policy._now = 0.0
    policy._time_fn = lambda: policy._now
    dead_key = _key(b"dead")
    live_key = _key(b"live")
    dead_block = ChunkStatus(0)
    live_block = ChunkStatus(1)
    dead_block.ref_cnt = 0
    live_block.ref_cnt = 0
    policy.insert(dead_key, dead_block)
    policy.insert(live_key, live_block)
    dead_ctx = ReqContext(
        req_id="d",
        session_id="dead",
        token_positions={dead_key: 0},
    )
    live_ctx = ReqContext(
        req_id="l",
        session_id="live",
        token_positions={live_key: 0},
    )
    policy.on_new_request(dead_ctx)
    policy.touch([dead_key], dead_ctx)
    policy._now = 0.1
    policy.notify_request_finished(dead_ctx)
    policy._now = 1.0
    policy.on_new_request(live_ctx)
    policy.touch([live_key], live_ctx)
    policy._now = 10.0
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == dead_key


def test_first_request_pin_holds_until_cpu_ttl():
    policy = FirstRequestPinCachePolicy(4, session_ttl_s=10)
    policy._time_fn = lambda: policy._now
    policy._now = 0.0
    stem = _key(b"stem-hash")
    other = _key(b"other")
    ctx = ReqContext(
        req_id="r0",
        session_id="s",
        priority=0,
        block_hashes=(b"stem-hash",),
    )
    policy.on_new_request(ctx)
    stem_block = ChunkStatus(0)
    other_block = ChunkStatus(1)
    stem_block.ref_cnt = 0
    other_block.ref_cnt = 0
    policy.insert(stem, stem_block)
    policy.insert(other, other_block)
    policy.note_stored([stem], ctx)
    policy._now = 0.1
    policy.notify_request_finished(ctx)
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == other
    policy._now = 11.0
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == stem


def test_session_ttl_follows_idle_moving_average():
    policy = PrefixPruningOnSessionsCachePolicy(
        4,
        first_tier_tokens=10_000,
        tier_step_tokens=10_000,
        max_tiers=4,
        session_ttl_s=0,
    )
    policy._ttl_scale = 2.0
    policy._now = 0.0
    policy._time_fn = lambda: policy._now
    fresh = _key(b"fresh")
    aged = _key(b"aged")
    for key, block_id in ((fresh, 0), (aged, 1)):
        block = ChunkStatus(block_id)
        block.ref_cnt = 0
        policy.insert(key, block)
    policy.on_new_request(
        ReqContext(req_id="a", session_id="aged", token_positions={aged: 0})
    )
    policy.touch([aged], ReqContext(req_id="a", session_id="aged", token_positions={aged: 0}))
    policy._now = 0.1
    policy.notify_request_finished(ReqContext(req_id="a", session_id="aged"))
    policy._now = 4.0
    policy.on_new_request(ReqContext(req_id="b", session_id="aged"))
    policy._now = 4.1
    policy.notify_request_finished(ReqContext(req_id="b", session_id="aged"))
    # Fresh session finished just now, so it is inside any positive limit.
    policy.on_new_request(
        ReqContext(req_id="c", session_id="fresh", token_positions={fresh: 0})
    )
    policy.touch(
        [fresh], ReqContext(req_id="c", session_id="fresh", token_positions={fresh: 0})
    )
    policy._now = 4.2
    policy.notify_request_finished(ReqContext(req_id="c", session_id="fresh"))
    # Make aged newer so, while both sessions are still live, LRU picks
    # the older fresh block.
    policy.touch(
        [aged], ReqContext(req_id="d", session_id="aged", token_positions={aged: 0})
    )
    # Aged limit is 2 * 3.9 = 7.8. At t=11.1 its idle is 7s, so aged is
    # still live and the older fresh block is the victim.
    policy._now = 11.1
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == fresh
    policy._now = 12.0
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == aged


def _ready(policy, key: bytes, block_id: int) -> None:
    block = ChunkStatus(block_id)
    block.ref_cnt = 0
    policy.insert(key, block)


def test_lru_tail_matches_cpu_lru_not_completion_order():
    """Blocks past the last position tier evict like CPU LRU.

    A store that finishes late must not become the newest block, and the
    tail of the older request goes before that request's head.
    """
    policy = PrefixPruningCachePolicy(
        8, first_tier_tokens=10, tier_step_tokens=10, max_tiers=1
    )
    old_head, old_tail = _key(b"oh"), _key(b"ot")
    new_head, new_tail = _key(b"nh"), _key(b"nt")
    for block_id, key in enumerate((old_head, old_tail, new_head, new_tail)):
        _ready(policy, key, block_id)
    policy.touch(
        [old_head, old_tail],
        ReqContext(
            req_id="old",
            token_positions={old_head: 100, old_tail: 200},
        ),
    )
    # Head store completes after the tail. That used to rewrite rank.
    policy.mark_evictable(old_tail)
    policy.mark_evictable(old_head)
    policy.touch(
        [new_head, new_tail],
        ReqContext(
            req_id="new",
            token_positions={new_head: 100, new_tail: 200},
        ),
    )
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == old_tail
    assert policy.get(old_head) is not None
    assert policy.get(new_head) is not None
    assert policy.get(new_tail) is not None


def test_position_tier_still_peels_the_earlier_token():
    policy = PrefixPruningCachePolicy(
        4, first_tier_tokens=1000, tier_step_tokens=1000, max_tiers=4
    )
    early, late = _key(b"early"), _key(b"late")
    _ready(policy, late, 0)
    _ready(policy, early, 1)
    policy.touch(
        [early, late],
        ReqContext(req_id="r", token_positions={early: 0, late: 50}),
    )
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == early


def test_session_lru_tail_matches_cpu_lru():
    policy = PrefixPruningOnSessionsCachePolicy(
        8,
        first_tier_tokens=10,
        tier_step_tokens=10,
        max_tiers=1,
        session_ttl_s=100,
    )
    old_tail, new_head = _key(b"ot"), _key(b"nh")
    _ready(policy, old_tail, 0)
    _ready(policy, new_head, 1)
    policy.touch(
        [old_tail],
        ReqContext(req_id="old", session_id="s", token_positions={old_tail: 500}),
    )
    policy.mark_evictable(old_tail)
    policy.touch(
        [new_head],
        ReqContext(req_id="new", session_id="s", token_positions={new_head: 500}),
    )
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == old_tail


def test_first_request_pin_evicts_unpinned_blocks_by_lru_tail():
    # With no recorded positions both blocks fall to the LRU tail, where the
    # tier policy still breaks ties by recency rank (oldest tail first).
    policy = FirstRequestPinCachePolicy(8, session_ttl_s=10)
    policy._time_fn = lambda: 0.0
    head, tail = _key(b"head"), _key(b"tail")
    _ready(policy, head, 0)
    _ready(policy, tail, 1)
    policy.touch([head, tail], ReqContext(req_id="r"))
    policy.mark_evictable(tail)
    policy.mark_evictable(head)
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == tail
    assert policy.get(head) is not None


def test_first_request_pin_peels_unpinned_prefix_tiers():
    # Non-pinned sessions must follow the prefix-tier peel so a reused suffix
    # survives and the connector can report a disjoint span.
    policy = FirstRequestPinCachePolicy(
        8, first_tier_tokens=1, tier_step_tokens=1, max_tiers=4, session_ttl_s=10
    )
    policy._time_fn = lambda: 0.0
    shallow, deep = _key(b"shallow"), _key(b"deep")
    shallow_block, deep_block = ChunkStatus(0), ChunkStatus(1)
    shallow_block.ref_cnt = 0
    deep_block.ref_cnt = 0
    policy.insert(shallow, shallow_block)
    policy.insert(deep, deep_block)
    ctx = ReqContext(
        req_id="r1",
        session_id="s1",
        priority=1,  # not the pinned priority-0 session
        token_positions={shallow: 0, deep: 100},
    )
    policy.touch([shallow, deep], ctx)
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == shallow
    assert policy.get(deep) is not None
    # The surviving suffix shows up as a disjoint span across the hole.
    assert policy.cached_spans([shallow, deep]) == [(1, 2)]


def test_first_request_pin_holds_stem_while_peeling_rest():
    # The pinned priority-0 stem is protected even though its position is the
    # shallowest tier; the rest of the session still peels prefix-first.
    policy = FirstRequestPinCachePolicy(
        8, first_tier_tokens=1, tier_step_tokens=1, max_tiers=4, session_ttl_s=10
    )
    policy._time_fn = lambda: policy._now
    policy._now = 0.0
    stem, mid = _key(b"stem"), _key(b"mid")
    stem_block, mid_block = ChunkStatus(0), ChunkStatus(1)
    stem_block.ref_cnt = 0
    mid_block.ref_cnt = 0
    ctx = ReqContext(
        req_id="r0",
        session_id="s0",
        priority=0,
        block_hashes=(b"stem",),
        token_positions={stem: 0, mid: 100},
    )
    policy.on_new_request(ctx)
    policy.insert(stem, stem_block)
    policy.insert(mid, mid_block)
    policy.note_stored([stem, mid], ctx)
    policy._now = 0.1
    policy.notify_request_finished(ctx)
    # While held, the shallow stem survives and the deeper block is the victim.
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == mid
    assert policy.get(stem) is not None
    # After the TTL the stem is an ordinary tier victim again.
    policy._now = 11.0
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert evicted[0][0] == stem


def test_prefix_spans_skip_a_hole():
    policy = PrefixPruningCachePolicy(4)
    a = _key(b"a")
    b = _key(b"b")
    c = _key(b"c")
    a_block = ChunkStatus(0)
    c_block = ChunkStatus(1)
    a_block.ref_cnt = 0
    c_block.ref_cnt = 0
    policy.insert(a, a_block)
    policy.insert(c, c_block)
    spans = policy.cached_spans([a, b, c])
    assert spans == [(0, 1), (2, 3)]
