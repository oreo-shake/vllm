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
    assert state.ewa_gap == 3.0
    assert tracker.adaptive_ttl(state) == 6.0

    tracker.on_finish("gold", 5.0)
    tracker.on_arrival("gold", 15.0, priority=0)
    assert tracker.adaptive_ttl(state) == 10.0


def test_non_gold_session_does_not_use_adaptive_ttl():
    tracker = SessionLivenessTracker(session_ttl_limit_s=10.0, session_ttl_k=2.0)
    tracker.on_arrival("silver", 0.0, priority=1, first_request_hashes=("a",))
    state = tracker.states["silver"]
    assert tracker.adaptive_ttl(state) is None
    assert state.first_request_hashes == ()
