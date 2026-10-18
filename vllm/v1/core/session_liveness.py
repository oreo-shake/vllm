# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-level liveness for suffix-preserving eviction.

A session is the conversation / agent run that shares one ``session_id``
across turns. While it looks alive, freed KV stays on the prefix-first
tiers. When inter-turn idle exceeds a threshold, the session is predicted
dead and its remaining unique blocks are promoted for full eviction.

Two detectors:

* ``ewa`` — per-session exponential average of think times. Threshold is
  ``max(DEAD_TTL, ewa + k·std)`` so a couple of short thinks cannot pull
  the cut below the static floor.
* ``global`` — server-wide reservoir of every observed think time.
  Threshold is ``max(DEAD_TTL, quantile of recent gaps)``. One session's
  short thinks cannot make that session look dead; the bar is "longer
  than almost every think we have seen on this server."

Priority 0 can opt into a separate bound, ``max(TTL_limit, K · EWMA gap)``.
The gap is completion to the next arrival. ``TTL_limit <= 0`` leaves this
off, so the detectors above are unchanged. Priorities 1+ do not use the
cap; with the think-time floor also at 0 they are never predicted dead.

A request that omits ``priority`` is priority 0. ``pin_first_request``
is separate from prefix tiers and from the think-time detector. It is
on only when the priority-0 TTL cap is configured. Then that session's
first request keeps its block hashes until the session is predicted
dead. Priorities 1+ do not. The priority check is the extension point
for later policies.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Detector = Literal["ewa", "global"]


def resolve_session_id(
    body_session_id: str | None = None,
    headers: Mapping[str, str] | None = None,
    extra_args: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve session identity with the same precedence as upstream vLLM.

    Body ``session_id`` wins, then ``X-Session-ID``, then
    ``extra_args["session_id"]`` / ``vllm_xargs["session_id"]``.
    """
    if isinstance(body_session_id, str) and body_session_id:
        return body_session_id
    if headers is not None:
        for key in ("X-Session-ID", "x-session-id"):
            value = headers.get(key)
            if isinstance(value, str) and value:
                return value
    if extra_args is not None:
        value = extra_args.get("session_id")
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class SessionLivenessState:
    """Per-session row. Keyed by ``session_id`` in the tracker table.

    ``first_arrival`` is the registering request. ``last_finish`` is the
    latest completion. ``n_requests`` counts arrivals. ``ewa_gap`` is the
    completion-to-next-arrival EWMA. ``first_request_hashes`` is the
    priority-0 prefix captured at registration. ``stale`` means idle
    expiry cleared advisory protection until a later arrival revives it.
    """

    last_finish: float | None = None
    last_arrival: float | None = None
    first_arrival: float | None = None
    n_requests: int = 0
    n_gaps: int = 0
    ewa_gap: float = 0.0
    ewa_gap_sq: float = 0.0
    predicted_dead: bool = False
    stale: bool = False
    priority: int = 0
    first_request_hashes: tuple[Any, ...] = ()

    @property
    def std(self) -> float:
        var = max(0.0, self.ewa_gap_sq - self.ewa_gap * self.ewa_gap)
        return math.sqrt(var)


class SessionLivenessTracker:
    """Think-time detectors: per-session EWA or server-wide quantile."""

    def __init__(
        self,
        min_ttl_s: float = 0.0,
        ewa_alpha: float = 0.3,
        dead_k: float = 3.0,
        min_gaps: int = 2,
        detector: Detector = "ewa",
        quantile: float = 0.95,
        gap_window: int = 256,
        session_ttl_limit_s: float = 0.0,
        session_ttl_k: float = 1.0,
        pin_first_request: bool = False,
        observe_idle: bool = False,
    ) -> None:
        if min_ttl_s < 0:
            raise ValueError(f"min_ttl_s must be >= 0, got {min_ttl_s}")
        if session_ttl_limit_s < 0:
            raise ValueError(
                "session_ttl_limit_s must be >= 0, "
                f"got {session_ttl_limit_s}"
            )
        if session_ttl_k < 0:
            raise ValueError(f"session_ttl_k must be >= 0, got {session_ttl_k}")
        if not 0.0 < ewa_alpha <= 1.0:
            raise ValueError(f"ewa_alpha must be in (0, 1], got {ewa_alpha}")
        if dead_k < 0:
            raise ValueError(f"dead_k must be >= 0, got {dead_k}")
        if min_gaps < 0:
            raise ValueError(f"min_gaps must be >= 0, got {min_gaps}")
        if detector not in ("ewa", "global"):
            raise ValueError(f"detector must be 'ewa' or 'global', got {detector}")
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        if gap_window < 1:
            raise ValueError(f"gap_window must be >= 1, got {gap_window}")
        self.min_ttl_s = min_ttl_s
        self.ewa_alpha = ewa_alpha
        self.dead_k = dead_k
        self.min_gaps = min_gaps
        self.detector: Detector = detector
        self.quantile = quantile
        self.gap_window = gap_window
        self.session_ttl_limit_s = session_ttl_limit_s
        self.session_ttl_k = session_ttl_k
        self.pin_first_request = pin_first_request
        # Record completion-to-arrival gaps without changing eviction.
        # Used when this tier is LRU but another tier's policy needs the
        # session's idle moving average.
        self.observe_idle = observe_idle
        self.states: dict[str, SessionLivenessState] = {}
        self._global_gaps: deque[float] = deque(maxlen=gap_window)

    @property
    def enabled(self) -> bool:
        return (
            self.min_ttl_s > 0
            or self.session_ttl_limit_s > 0
            or self.pin_first_request
            or self.observe_idle
        )

    def on_arrival(
        self,
        session_id: str | None,
        now: float,
        priority: int = 0,
        first_request_hashes: tuple[Any, ...] = (),
    ) -> None:
        if not session_id or not self.enabled:
            return
        state = self.states.setdefault(session_id, SessionLivenessState())
        # Gap is completion → next arrival, including the gap that revives
        # a stale session. A turn that arrives before the previous finish
        # contributes no sample.
        if state.last_finish is not None:
            gap = max(0.0, now - state.last_finish)
            self._observe_gap(state, gap)
        if state.first_arrival is None:
            state.first_arrival = now
            state.priority = priority
            if (
                priority == 0
                and (self.session_ttl_limit_s > 0 or self.pin_first_request)
                and first_request_hashes
            ):
                state.first_request_hashes = first_request_hashes
        state.n_requests += 1
        state.last_arrival = now
        # A new arrival revives the session and its advisory prefix.
        state.predicted_dead = False
        state.stale = False

    def on_finish(
        self,
        session_id: str | None,
        now: float,
        block_hashes: tuple[Any, ...] = (),
    ) -> None:
        if not session_id or not self.enabled:
            return
        state = self.states.setdefault(session_id, SessionLivenessState())
        state.last_finish = now
        # Arrival usually has the prompt prefix. If that snapshot was empty,
        # the first completion can still seal it before blocks are freed.
        if (
            (self.session_ttl_limit_s > 0 or self.pin_first_request)
            and state.priority == 0
            and state.n_requests == 1
            and state.first_arrival is not None
            and not state.first_request_hashes
            and block_hashes
        ):
            state.first_request_hashes = block_hashes

    def adaptive_ttl(self, state: SessionLivenessState) -> float | None:
        """Return ``max(TTL_limit, K · EWMA)`` for an active priority-0 session.

        ``None`` means this session stays on the ordinary think-time
        detector (priorities 1+, or the bound left at 0). With no
        observed gap yet, the tier bound itself is the TTL. After a gap,
        an outlier average raises the TTL above that bound.
        """
        if self.session_ttl_limit_s <= 0 or state.priority != 0:
            return None
        if state.n_gaps == 0:
            return self.session_ttl_limit_s
        return max(self.session_ttl_limit_s, self.session_ttl_k * state.ewa_gap)

    def first_request_hashes_for(
        self, session_id: str | None, now: float
    ) -> tuple[Any, ...]:
        """Advisory first-request hashes, or empty when protection is off.

        Off unless the priority-0 cap is configured, the session is
        priority 0, and it has not gone stale. Idle expiry marks the
        session stale and drops the metadata. A later arrival revives it.
        """
        if not session_id:
            return ()
        if self.session_ttl_limit_s <= 0 and not self.pin_first_request:
            return ()
        state = self.states.get(session_id)
        if (
            state is None
            or state.priority != 0
            or not state.first_request_hashes
        ):
            return ()
        if state.stale or self.is_dead(session_id, now):
            state.stale = True
            state.predicted_dead = True
            return ()
        return state.first_request_hashes

    def threshold(self, state: SessionLivenessState) -> float:
        adaptive_ttl = self.adaptive_ttl(state)
        if adaptive_ttl is not None:
            return adaptive_ttl
        # The think-time detector stays off when its floor is 0. Enabling
        # only the priority-0 cap must not mark priorities 1+ dead.
        if self.min_ttl_s <= 0:
            return math.inf
        if self.detector == "global":
            if len(self._global_gaps) < self.min_gaps:
                return self.min_ttl_s
            return max(self.min_ttl_s, self._global_quantile())
        if state.n_gaps < self.min_gaps:
            return self.min_ttl_s
        return max(self.min_ttl_s, state.ewa_gap + self.dead_k * state.std)

    def idle_limit(
        self, session_id: str | None, scale: float, floor_s: float = 0.0
    ) -> float | None:
        """``scale`` times this session's idle-gap EWMA.

        ``None`` means the session has no gap yet and no positive floor,
        so a caller must not treat it as expired. A positive ``floor_s``
        is the limit until the first gap, and a lower bound after that.
        """
        if not session_id:
            return None
        state = self.states.get(session_id)
        if state is None or state.n_gaps == 0:
            return floor_s if floor_s > 0 else None
        limit = scale * state.ewa_gap
        if floor_s > 0:
            limit = max(floor_s, limit)
        return limit

    def idle_expired(
        self,
        session_id: str | None,
        now: float,
        scale: float,
        floor_s: float = 0.0,
    ) -> bool:
        """True when the session has been idle for ``scale * ewa`` seconds.

        A turn still in flight is not expired. Observation-only trackers
        use this instead of ``is_dead``, which stays false while no dead
        detector is configured.
        """
        if not session_id:
            return False
        state = self.states.get(session_id)
        if state is None or state.last_finish is None:
            return False
        if state.last_arrival is not None and state.last_arrival >= state.last_finish:
            return False
        limit = self.idle_limit(session_id, scale, floor_s)
        if limit is None:
            return False
        return (now - state.last_finish) >= limit

    def is_dead(self, session_id: str, now: float) -> bool:
        if not self.enabled:
            return False
        state = self.states.get(session_id)
        if state is None or state.last_finish is None:
            return False
        # A turn is in flight if it arrived after the last finish.
        if state.last_arrival is not None and state.last_arrival >= state.last_finish:
            return False
        return (now - state.last_finish) >= self.threshold(state)

    def dead_session_ids(self, now: float) -> list[str]:
        if not self.enabled:
            return []
        return [sid for sid in self.states if self.is_dead(sid, now)]

    def mark_promoted(self, session_ids: list[str]) -> None:
        for sid in session_ids:
            state = self.states.get(sid)
            if state is not None:
                state.predicted_dead = True
                state.stale = True

    def _observe_gap(self, state: SessionLivenessState, gap: float) -> None:
        self._global_gaps.append(gap)
        if state.n_gaps == 0:
            state.ewa_gap = gap
            state.ewa_gap_sq = gap * gap
        else:
            a = self.ewa_alpha
            state.ewa_gap = a * gap + (1.0 - a) * state.ewa_gap
            state.ewa_gap_sq = a * gap * gap + (1.0 - a) * state.ewa_gap_sq
        state.n_gaps += 1

    def _global_quantile(self) -> float:
        if not self._global_gaps:
            return self.min_ttl_s
        xs = sorted(self._global_gaps)
        idx = int((len(xs) - 1) * self.quantile)
        return xs[idx]
