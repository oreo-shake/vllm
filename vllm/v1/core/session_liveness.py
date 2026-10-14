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
    last_finish: float | None = None
    last_arrival: float | None = None
    n_gaps: int = 0
    ewa_gap: float = 0.0
    ewa_gap_sq: float = 0.0
    predicted_dead: bool = False
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
        self.states: dict[str, SessionLivenessState] = {}
        self._global_gaps: deque[float] = deque(maxlen=gap_window)

    @property
    def enabled(self) -> bool:
        return self.min_ttl_s > 0 or self.session_ttl_limit_s > 0

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
        if state.last_finish is not None:
            gap = max(0.0, now - state.last_finish)
            self._observe_gap(state, gap)
        if state.last_arrival is None:
            state.priority = priority
            if priority == 0:
                state.first_request_hashes = first_request_hashes
        state.last_arrival = now
        state.predicted_dead = False

    def on_finish(self, session_id: str | None, now: float) -> None:
        if not session_id or not self.enabled:
            return
        state = self.states.setdefault(session_id, SessionLivenessState())
        state.last_finish = now

    def adaptive_ttl(self, state: SessionLivenessState) -> float | None:
        """Return the capped local TTL for an active priority-0 session."""
        if self.session_ttl_limit_s <= 0 or state.priority != 0:
            return None
        if state.n_gaps == 0:
            return self.session_ttl_limit_s
        return min(self.session_ttl_limit_s, self.session_ttl_k * state.ewa_gap)

    def threshold(self, state: SessionLivenessState) -> float:
        adaptive_ttl = self.adaptive_ttl(state)
        if adaptive_ttl is not None:
            return adaptive_ttl
        if self.detector == "global":
            if len(self._global_gaps) < self.min_gaps:
                return self.min_ttl_s
            return max(self.min_ttl_s, self._global_quantile())
        if state.n_gaps < self.min_gaps:
            return self.min_ttl_s
        return max(self.min_ttl_s, state.ewa_gap + self.dead_k * state.std)

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
