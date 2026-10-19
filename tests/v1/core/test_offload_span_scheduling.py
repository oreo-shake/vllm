# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CPU offload span clip/requeue under async scheduling.

prefix_pruning* policies record disjoint CPU runs in ``_offload_spans``.
Under async scheduling, ``num_computed_tokens`` advances when the last hole
chunk is scheduled while tokens are still in flight. The running-path clip
must pause (not drop the span) until requeue can load from CPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


def _scheduler_with_spans(
    request_id: str, spans: list[tuple[int, int]]
) -> tuple[Scheduler, MagicMock]:
    sched = object.__new__(Scheduler)
    sched._offload_spans = {request_id: list(spans)}
    sched.running = []
    sched.waiting = MagicMock()
    return sched, sched.waiting


def _request(
    request_id: str,
    *,
    num_computed_tokens: int,
    num_in_flight_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_computed_tokens=num_computed_tokens,
        num_in_flight_tokens=num_in_flight_tokens,
        status=RequestStatus.RUNNING,
    )


def test_clip_pause_keeps_span_when_cursor_inside_run():
    """Running path must not discard a CPU span while hole tokens are in flight."""
    req_id = "r0"
    sched, _ = _scheduler_with_spans(req_id, [(1024, 4096)])
    request = _request(req_id, num_computed_tokens=1024, num_in_flight_tokens=256)

    clipped = sched._clip_to_offload_span(
        request, num_new_tokens=2048, num_computed=1024, pause=True
    )

    assert clipped == 0
    assert sched._offload_spans[req_id] == [(1024, 4096)]


def test_clip_without_pause_drops_span_for_recompute():
    """Waiting path (pause=False) still drops a stranded span and recomputes."""
    req_id = "r0"
    sched, _ = _scheduler_with_spans(req_id, [(1024, 4096)])
    request = _request(req_id, num_computed_tokens=1024)

    clipped = sched._clip_to_offload_span(
        request, num_new_tokens=2048, num_computed=1024, pause=False
    )

    assert clipped == 2048
    assert req_id not in sched._offload_spans


def test_clip_limits_to_hole_before_span():
    req_id = "r0"
    sched, _ = _scheduler_with_spans(req_id, [(1024, 4096)])
    request = _request(req_id, num_computed_tokens=0)

    clipped = sched._clip_to_offload_span(
        request, num_new_tokens=8192, num_computed=0, pause=True
    )

    assert clipped == 1024
    assert sched._offload_spans[req_id] == [(1024, 4096)]


def test_requeue_waits_for_in_flight_then_moves_to_waiting():
    req_id = "r0"
    sched, waiting = _scheduler_with_spans(req_id, [(1024, 4096)])
    request = _request(req_id, num_computed_tokens=1024, num_in_flight_tokens=128)
    sched.running = [request]

    assert sched._requeue_for_offload_span(request, 0) is False
    assert request in sched.running
    assert request.status == RequestStatus.RUNNING

    request.num_in_flight_tokens = 0
    assert sched._requeue_for_offload_span(request, 0) is True
    assert sched.running == []
    assert request.status == RequestStatus.WAITING
    waiting.prepend_request.assert_called_once_with(request)
    assert sched._offload_spans[req_id] == [(1024, 4096)]


def test_pending_offload_span_tokens_sums_deferred_runs():
    req_id = "r0"
    sched, _ = _scheduler_with_spans(req_id, [(1024, 2048), (4096, 8192)])
    assert sched._pending_offload_span_tokens(req_id) == (2048 - 1024) + (
        8192 - 4096
    )
    assert sched._pending_offload_span_tokens("missing") == 0


def test_match_offload_span_discards_rediscovered_deferred_runs():
    """Mid-request lookup must not re-ingest later runs into _offload_spans."""
    req_id = "r0"
    sched, _ = _scheduler_with_spans(req_id, [(1024, 2048), (4096, 8192)])
    request = _request(req_id, num_computed_tokens=1024)

    rediscovered = [(4096, 8192)]  # same later run already in the plan
    connector = MagicMock()
    connector.get_num_new_matched_tokens.return_value = (1024, False)
    connector.take_offload_spans.return_value = list(rediscovered)
    sched.connector = connector

    ext_tokens, load_async = sched._match_offload_span(request, 1024)

    assert (ext_tokens, load_async) == (1024, False)
    connector.take_offload_spans.assert_called_once_with(request)
    # Current run consumed; remaining plan unchanged (not duplicated).
    assert sched._offload_spans[req_id] == [(4096, 8192)]
    assert sched._pending_offload_span_tokens(req_id) == 8192 - 4096
