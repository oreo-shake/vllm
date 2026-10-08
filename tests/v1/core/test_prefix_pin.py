# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage-cliff detector and pin-hold for automatic system-prompt pinning."""

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.prefix_pin import (
    SystemPromptPinDetector,
    cliff_length,
    coverage_curve,
    prefix_share,
)


def _h(tag: str) -> str:
    return tag


def _sys_then_unique(sys_len: int, unique: str, extra: int = 2) -> tuple[str, ...]:
    return tuple(_h(f"S{i}") for i in range(sys_len)) + (unique,) + tuple(
        _h(f"{unique}x{i}") for i in range(extra)
    )


def test_coverage_curve_is_monotone_and_finds_cliff():
    obs = [_sys_then_unique(4, f"u{i}") for i in range(8)]
    mode, coverage = coverage_curve(obs, max_blocks=16)
    assert mode[:4] == ("S0", "S1", "S2", "S3")
    assert coverage[0] == 1.0
    assert coverage[3] == 1.0
    assert coverage[4] < 0.3
    assert all(coverage[i] >= coverage[i + 1] for i in range(len(coverage) - 1))
    assert cliff_length(coverage, share=0.75, drop=0.2) == 4


def test_cliff_length_stops_below_share():
    # Slow decay, no sharp drop: pin until coverage leaves the plateau.
    coverage = (1.0, 0.9, 0.8, 0.7, 0.6)
    assert cliff_length(coverage, share=0.75, drop=0.2) == 3


def test_no_shared_first_block_is_not_a_system_prompt():
    obs = [(f"t{i}", "x") for i in range(8)]
    mode, coverage = coverage_curve(obs, max_blocks=8)
    assert coverage[0] == 1.0 / 8
    assert cliff_length(coverage, share=0.75, drop=0.2) == 0


def test_detector_needs_n_unique_first_turns_and_confirm():
    det = SystemPromptPinDetector(
        min_requests=4,
        share=0.75,
        drop=0.2,
        max_pin_blocks=16,
        confirm=2,
        window=16,
    )
    for i in range(4):
        assert det.observe(f"s{i}", _sys_then_unique(3, f"u{i}")) is None
    # 4 unique first-turns: candidate exists, confirm=1, not committed.
    assert det.pinned_hashes == ()
    snap = det.observe("s4", _sys_then_unique(3, "u4"))
    assert snap is not None
    assert snap.hashes == ("S0", "S1", "S2")
    assert snap.cliff == 3
    assert snap.n_obs == 5


def test_detector_ignores_later_turns_of_same_session():
    det = SystemPromptPinDetector(
        min_requests=3, share=0.7, drop=0.2, confirm=1, window=16
    )
    # Same session, growing history: must not pin the whole chat.
    long = ("S0", "S1", "t0", "t1", "t2", "t3")
    assert det.observe("s0", ("S0", "S1", "t0")) is None
    assert det.observe("s0", long) is None
    det.observe("s1", ("S0", "S1", "a"))
    snap = det.observe("s2", ("S0", "S1", "b"))
    assert snap is not None
    assert snap.hashes == ("S0", "S1")


def test_detector_two_tenants_cluster_independently():
    det = SystemPromptPinDetector(
        min_requests=3,
        share=0.7,
        drop=0.2,
        confirm=1,
        window=32,
        max_clusters=2,
    )
    for i in range(3):
        det.observe(f"a{i}", ("A0", "A1", f"au{i}"))
    for i in range(3):
        snap = det.observe(f"b{i}", ("B0", "B1", f"bu{i}"))
    assert snap is not None
    assert set(snap.hashes) == {"A0", "A1", "B0", "B1"}
    assert snap.clusters == 2


def test_detector_unpins_when_share_collapses():
    det = SystemPromptPinDetector(
        min_requests=3,
        share=0.7,
        drop=0.2,
        confirm=1,
        window=6,
        unpin_share=0.4,
    )
    for i in range(3):
        snap = det.observe(f"s{i}", ("S0", "S1", f"u{i}"))
    assert snap is not None
    assert snap.hashes == ("S0", "S1")
    # Window=6: three matching then three different first-blocks → share 0.5
    # then after more, matching prefix leaves the window.
    # Distinct first-blocks so no new cluster can reach min_requests.
    later = None
    for i in range(6):
        later = det.observe(f"x{i}", (f"Z{i}", "zz"))
    assert later is not None
    assert later.hashes == ()


def test_prefix_share():
    prefix = ("S0", "S1")
    obs = [("S0", "S1", "a"), ("S0", "S1", "b"), ("S0", "other")]
    assert prefix_share(prefix, obs) == 2 / 3


def _bh(tag: str) -> BlockHash:
    raw = tag.encode("ascii")
    return BlockHash(raw + b"\x00" * (32 - len(raw)))


def _insert(pool: BlockPool, block, tag: str) -> BlockHash:
    h = _bh(tag)
    pool._insert_block_hash(make_block_hash_with_group_id(h, 0), block, num_tokens=16)
    return h


def test_pin_hold_keeps_block_off_free_queue():
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16)
    free_before = pool.get_num_free_blocks()
    blocks = pool.get_new_blocks(3)
    h0 = _insert(pool, blocks[0], "sys0")
    h1 = _insert(pool, blocks[1], "sys1")
    _insert(pool, blocks[2], "uniq")
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == free_before

    held, released = pool.set_pinned_hashes((h0, h1))
    assert held == 2
    assert released == 0
    assert blocks[0].pinned and blocks[1].pinned
    assert not blocks[2].pinned
    assert pool.get_num_free_blocks() == free_before - 2

    stolen = pool.get_new_blocks(free_before - 2)
    assert blocks[0] not in stolen
    assert blocks[1] not in stolen
    assert blocks[0].ref_cnt == 1
    pool.free_blocks(stolen)


def test_pin_after_last_user_free_still_holds():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(1)
    h0 = _insert(pool, blocks[0], "sys0")
    pool.set_pinned_hashes((h0,))
    assert blocks[0].ref_cnt == 2
    pool.free_blocks(blocks)
    assert blocks[0].ref_cnt == 1
    assert blocks[0].pinned
    assert blocks[0] not in pool.free_block_queue.get_all_free_blocks()


def test_unpin_returns_block_to_free_queue():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    free_before = pool.get_num_free_blocks()
    blocks = pool.get_new_blocks(1)
    h0 = _insert(pool, blocks[0], "sys0")
    pool.free_blocks(blocks)
    pool.set_pinned_hashes((h0,))
    released = pool.clear_pinned_hashes()
    assert released == 1
    assert not blocks[0].pinned
    assert pool.get_num_free_blocks() == free_before


def test_new_cached_block_of_pinned_hash_is_held():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    h0 = _bh("sys0")
    pool.set_pinned_hashes((h0,))
    blocks = pool.get_new_blocks(1)
    pool._insert_block_hash(
        make_block_hash_with_group_id(h0, 0), blocks[0], num_tokens=16
    )
    assert blocks[0].pinned
    assert blocks[0].ref_cnt == 2
    pool.free_blocks(blocks)
    assert blocks[0].ref_cnt == 1


def test_prefix_first_cannot_evict_pinned_system_prompt():
    pool = BlockPool(num_gpu_blocks=12, enable_caching=True, hash_block_size=16)
    pool.enable_prefix_first_eviction(num_tiers=2)
    free_before = pool.get_num_free_blocks()
    blocks = pool.get_new_blocks(4)
    hashes = [_insert(pool, block, f"p{i}") for i, block in enumerate(blocks)]
    pool.free_blocks_tiered(blocks, first_tier_len=2, tier_len=2)
    # Leading blocks sit in tier 0 — exactly what prefix-first would peel.
    pool.set_pinned_hashes(tuple(hashes[:2]))
    assert blocks[0].pinned and blocks[1].pinned
    assert pool.get_num_free_blocks() == free_before - 2
    taken = pool.get_new_blocks(free_before - 2)
    assert blocks[0] not in taken and blocks[1] not in taken
    pool.free_blocks(taken)


def test_reset_prefix_cache_releases_pins():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(1)
    h0 = _insert(pool, blocks[0], "sys0")
    pool.free_blocks(blocks)
    pool.set_pinned_hashes((h0,))
    assert pool.reset_prefix_cache()
    assert not pool.pinned_hashes
    assert not blocks[0].pinned
    assert blocks[0].block_hash is None


if __name__ == "__main__":
    tests = [
        test_coverage_curve_is_monotone_and_finds_cliff,
        test_cliff_length_stops_below_share,
        test_no_shared_first_block_is_not_a_system_prompt,
        test_detector_needs_n_unique_first_turns_and_confirm,
        test_detector_ignores_later_turns_of_same_session,
        test_detector_two_tenants_cluster_independently,
        test_detector_unpins_when_share_collapses,
        test_prefix_share,
        test_pin_hold_keeps_block_off_free_queue,
        test_pin_after_last_user_free_still_holds,
        test_unpin_returns_block_to_free_queue,
        test_new_cached_block_of_pinned_hash_is_held,
        test_prefix_first_cannot_evict_pinned_system_prompt,
        test_reset_prefix_cache_releases_pins,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
            raise
    print(f"{len(tests) - failed}/{len(tests)} passed")
