from vllm.v1.core.session_liveness import SessionLivenessTracker


def test_priority_zero_adaptive_ttl_is_capped_and_uses_completion_gap():
    tracker = SessionLivenessTracker(
        session_ttl_limit_s=10.0,
        session_ttl_k=2.0,
        ewa_alpha=1.0,
    )
    tracker.on_arrival("gold", 0.0, priority=0, first_request_hashes=("a",))
    tracker.on_finish("gold", 1.0)
    tracker.on_arrival("gold", 4.0, priority=0)

    state = tracker.states["gold"]
    assert state.first_arrival == 0.0
    assert state.n_requests == 2
    assert state.last_finish == 1.0
    assert state.priority == 0
    assert state.ewa_gap == 3.0
    assert state.first_request_hashes == ("a",)
    assert tracker.adaptive_ttl(state) == 6.0
    assert tracker.first_request_hashes_for("gold", 4.0) == ("a",)

    tracker.on_finish("gold", 5.0)
    # Idle from the completion at t=5. Cap is min(10, 2*3) = 6.
    assert not tracker.is_dead("gold", 10.9)
    assert tracker.is_dead("gold", 11.0)
    assert tracker.first_request_hashes_for("gold", 11.0) == ()
    assert state.stale

    # A later arrival revives the session and keeps the registered prefix.
    tracker.on_arrival("gold", 15.0, priority=0, first_request_hashes=("zzz",))
    assert tracker.adaptive_ttl(state) == 10.0
    assert not state.stale
    assert state.n_requests == 3
    assert state.first_request_hashes == ("a",)
    assert tracker.first_request_hashes_for("gold", 15.0) == ("a",)


def test_non_gold_session_does_not_use_adaptive_ttl():
    tracker = SessionLivenessTracker(session_ttl_limit_s=10.0, session_ttl_k=2.0)
    tracker.on_arrival("silver", 0.0, priority=1, first_request_hashes=("a",))
    tracker.on_finish("silver", 1.0)
    state = tracker.states["silver"]
    assert tracker.adaptive_ttl(state) is None
    assert state.first_request_hashes == ()
    assert state.n_requests == 1
    assert state.priority == 1
    # Cap-only mode must not predict priorities 1+ dead.
    assert not tracker.is_dead("silver", 10_000.0)
    assert tracker.first_request_hashes_for("silver", 10_000.0) == ()


def test_cap_defaults_off_and_rejects_bad_values():
    off = SessionLivenessTracker()
    assert not off.enabled
    off.on_arrival("s", 0.0, priority=0, first_request_hashes=("a",))
    assert "s" not in off.states

    try:
        SessionLivenessTracker(session_ttl_limit_s=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative TTL limit must be rejected")

    try:
        SessionLivenessTracker(session_ttl_k=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative TTL multiplier must be rejected")


def test_first_completion_seals_empty_prefix_snapshot():
    tracker = SessionLivenessTracker(session_ttl_limit_s=8.0, session_ttl_k=1.0)
    tracker.on_arrival("gold", 0.0, priority=0, first_request_hashes=())
    tracker.on_finish("gold", 1.0, block_hashes=("prompt",))
    assert tracker.states["gold"].first_request_hashes == ("prompt",)
    assert tracker.first_request_hashes_for("gold", 1.0) == ("prompt",)

    # A second completion must not replace the registered prefix.
    tracker.on_arrival("gold", 2.0, priority=0, first_request_hashes=("other",))
    tracker.on_finish("gold", 3.0, block_hashes=("other",))
    assert tracker.states["gold"].first_request_hashes == ("prompt",)
