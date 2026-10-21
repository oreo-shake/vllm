# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU eviction policies that sit beside LRU and ARC.

``lru`` and ``arc`` are not modified. These classes are selected only when
``eviction_policy`` names them.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence

from typing_extensions import override

from vllm import envs
from vllm.v1.core.session_liveness import SessionLivenessTracker
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
)

UNKNOWN_POSITION = 1 << 30


def resolve_cpu_session_ttl_s() -> float:
    """CPU tier bound used in ``max(K * idle EWMA, bound)``.

    ``VLLM_CPU_SESSION_TTL_S`` only. It is not derived from the GPU TTL.
    ``VLLM_CPU_SESSION_TTL_K`` is the factor on the moving average.
    """
    explicit = envs.VLLM_CPU_SESSION_TTL_S
    if explicit:
        return float(explicit)
    return 0.0


class _TierScorer:
    def __init__(
        self, first_tier_tokens: int, tier_step_tokens: int, max_tiers: int
    ) -> None:
        if first_tier_tokens < 1 or tier_step_tokens < 1 or max_tiers < 0:
            raise ValueError("CPU prefix tier sizes must be positive")
        self.first_tier_tokens = first_tier_tokens
        self.tier_step_tokens = tier_step_tokens
        self.max_tiers = max_tiers

    def score(
        self, position: int, rank: int, key: OffloadKey, is_dead: bool
    ) -> tuple[int, int, int, int, OffloadKey]:
        if position >= UNKNOWN_POSITION:
            tier = UNKNOWN_POSITION
        elif position <= self.first_tier_tokens:
            tier = 0
        else:
            tier = 1 + (position - self.first_tier_tokens - 1) // self.tier_step_tokens
        # Past the last position tier, or with no recorded position, the block
        # is in the LRU tail. That tail uses the same rank as CPU LRU: a
        # smaller rank is the older tail, not the block that finished storing
        # first. Position tiers still peel the earlier token first.
        in_lru_tail = position >= UNKNOWN_POSITION or (
            bool(self.max_tiers) and tier >= self.max_tiers
        )
        if is_dead:
            return (0, position, rank, 0, key)
        if in_lru_tail:
            return (2, 0, rank, 0, key)
        return (1, tier, position, rank, key)


def _session_id(req_context: ReqContext) -> str | None:
    if req_context.session_id:
        return req_context.session_id
    params = req_context.kv_transfer_params
    if params:
        value = params.get("session_id")
        if isinstance(value, str) and value:
            return value
    return None


def _priority(req_context: ReqContext) -> int:
    if req_context.priority is None:
        return 0
    return req_context.priority




class PrefixPruningCachePolicy(CachePolicy):
    """Peel shallow CPU tiers before the LRU tail. No session identity."""

    # Connector touch supplies start-token positions and refreshes LRU order.
    # Matches the older prefix_pruning / prefix_pruning_on_sessions contract.
    refreshes_recency_from_connector = True
    supports_disjoint_spans = True

    def __init__(
        self,
        cache_capacity: int,
        first_tier_tokens: int = 16384,
        tier_step_tokens: int = 16384,
        max_tiers: int = 8,
        **_ignored: object,
    ) -> None:
        super().__init__(cache_capacity)
        self._scorer = _TierScorer(first_tier_tokens, tier_step_tokens, max_tiers)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._ranks: dict[OffloadKey, int] = {}
        self._positions: dict[OffloadKey, int] = {}
        self._sessions: dict[OffloadKey, str] = {}
        self._session_keys: dict[str, set[OffloadKey]] = defaultdict(set)
        self._evictable: set[OffloadKey] = set()
        self._next_rank = 0
        self._tracker: SessionLivenessTracker | None = None
        self._shared_tracker: SessionLivenessTracker | None = None
        self._ttl_scale = envs.VLLM_CPU_SESSION_TTL_K
        self._ttl_floor = 0.0
        self._time_fn = time.monotonic

    def _note_position(self, key: OffloadKey, req_context: ReqContext) -> None:
        # Tier peel uses start-token positions from connector touch
        # (index * tokens_per_chunk). Do not fall back to end-token
        # get_offload_key_position — that can push the chunk at a tier
        # boundary into a deeper tier and change eviction order.
        positions = req_context.token_positions
        if positions is not None and key in positions:
            self._positions[key] = positions[key]

    def _note_session(self, key: OffloadKey, session_id: str | None) -> None:
        if not session_id:
            return
        old = self._sessions.get(key)
        if old and old != session_id:
            self._session_keys[old].discard(key)
        self._sessions[key] = session_id
        self._session_keys[session_id].add(key)

    def _forget(self, key: OffloadKey) -> None:
        session_id = self._sessions.pop(key, None)
        if session_id:
            keys = self._session_keys[session_id]
            keys.discard(key)
            if not keys:
                self._session_keys.pop(session_id, None)
        self._positions.pop(key, None)
        self._ranks.pop(key, None)
        self._evictable.discard(key)

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    def _bump(self, key: OffloadKey) -> None:
        self._next_rank += 1
        self._ranks[key] = self._next_rank

    def _refresh_lru_order(self, keys: Iterable[OffloadKey]) -> None:
        # Same finish order as CPU LRU: the first key ends newest, so the
        # LRU tail drops the end of an older request before its head.
        for key in reversed(list(keys)):
            if key in self._ranks:
                self._bump(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.chunks[key] = chunk
        self._bump(key)
        if chunk.ref_cnt == 0:
            self._evictable.add(key)

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self._forget(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        session_id = _session_id(req_context)
        present: list[OffloadKey] = []
        for key in keys:
            if key not in self.chunks:
                continue
            self._note_position(key, req_context)
            self._note_session(key, session_id)
            present.append(key)
        self._refresh_lru_order(present)

    @override
    def note_finished_keys(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        # Used when refreshes_recency_from_connector is false. Prefix policies
        # keep refreshes true so connector touch owns mid-request order; this
        # remains for API parity with the older branch.
        del req_context
        self._refresh_lru_order(keys)

    def note_stored(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        # Bind session (and start-token position when touch already filled
        # token_positions) on the store path. Tier positions still come from
        # connector touch — same as the older branch.
        session_id = _session_id(req_context)
        for key in keys:
            if key not in self.chunks:
                continue
            self._note_position(key, req_context)
            self._note_session(key, session_id)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        # Newer manager passes key groups; prefix policies ignore them for
        # recency. Connector touch already refreshed LRU (older behavior).
        # Only update the session liveness tracker here.
        del key_groups, insertion_only_keys, reused_keys
        self.notify_request_finished(req_context)

    def attach_session_tracker(self, tracker: SessionLivenessTracker) -> None:
        """Use the GPU-side idle record so LRU-on-GPU still feeds this policy."""
        self._shared_tracker = tracker

    def _active_tracker(self) -> SessionLivenessTracker | None:
        return self._shared_tracker or self._tracker

    def on_new_request(self, req_context: ReqContext) -> None:
        # The shared tracker is updated by the scheduler. Recording here
        # again would count the same gap twice.
        if self._shared_tracker is not None or self._tracker is None:
            return
        self._tracker.on_arrival(
            _session_id(req_context),
            self._time_fn(),
            priority=_priority(req_context),
        )

    def notify_request_finished(self, req_context: ReqContext) -> None:
        if self._shared_tracker is not None or self._tracker is None:
            return
        self._tracker.on_finish(_session_id(req_context), self._time_fn())

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # A store that finishes late must not become newer than one inserted
        # earlier. Rank is the LRU order recorded at insert and at touch.
        self._evictable.add(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._evictable.discard(key)

    def _dead_sessions(self) -> set[str]:
        tracker = self._active_tracker()
        if tracker is None:
            return set()
        now = self._time_fn()
        return {
            session_id
            for session_id in tracker.states
            if tracker.idle_expired(
                session_id, now, self._ttl_scale, self._ttl_floor
            )
        }

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        dead = self._dead_sessions()
        candidates = []
        for key in self._evictable - protected:
            session_id = self._sessions.get(key)
            candidates.append(
                self._scorer.score(
                    self._positions.get(key, UNKNOWN_POSITION),
                    self._ranks.get(key, 0),
                    key,
                    session_id is not None and session_id in dead,
                )
            )
        if len(candidates) < n:
            return None
        evicted = []
        for *_, key in heapq.nsmallest(n, candidates):
            chunk = self.chunks.pop(key)
            self._forget(key)
            evicted.append((key, chunk))
        return evicted

    @override
    def clear(self) -> None:
        self.chunks.clear()
        self._ranks.clear()
        self._positions.clear()
        self._sessions.clear()
        self._session_keys.clear()
        self._evictable.clear()
        self._next_rank = 0


class PrefixPruningOnSessionsCachePolicy(PrefixPruningCachePolicy):
    """Prefix tiers plus a CPU-local dead-session wipe.

    The tracker is not the GPU tracker. Its floor is the CPU session TTL.
    """

    def __init__(
        self,
        cache_capacity: int,
        first_tier_tokens: int = 16384,
        tier_step_tokens: int = 16384,
        max_tiers: int = 8,
        session_ttl_s: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            cache_capacity,
            first_tier_tokens=first_tier_tokens,
            tier_step_tokens=tier_step_tokens,
            max_tiers=max_tiers,
        )
        del kwargs
        # Floor used only until the session has an idle-gap average.
        # After that the limit is max(floor, scale * EWMA).
        self._ttl_floor = (
            resolve_cpu_session_ttl_s() if session_ttl_s is None else session_ttl_s
        )
        self._ttl_scale = envs.VLLM_CPU_SESSION_TTL_K
        self._tracker = SessionLivenessTracker(observe_idle=True)


class FirstRequestPinCachePolicy(PrefixPruningCachePolicy):
    """Prefix tiers, plus a hold on a priority-0 session's first request.

    Non-pinned blocks follow the same position-tier peel as ``prefix_pruning``
    (shallow prefix tiers are evicted before the LRU tail), so reuse keeps the
    deep suffix cached and the connector reports the surviving run as a
    disjoint span. On top of that peel, the first request of a priority-0
    session is protected from eviction until the session has been idle for the
    CPU session TTL. Blocks still on GPU are not pinned here; a hash is
    protected only after it is stored. A missing ``priority`` is 0; priority 1
    and above are ordinary tier victims.
    """

    supports_disjoint_spans = True

    def __init__(
        self,
        cache_capacity: int,
        first_tier_tokens: int = 16384,
        tier_step_tokens: int = 16384,
        max_tiers: int = 8,
        session_ttl_s: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            cache_capacity,
            first_tier_tokens=first_tier_tokens,
            tier_step_tokens=tier_step_tokens,
            max_tiers=max_tiers,
        )
        del kwargs
        self._ttl_floor = (
            resolve_cpu_session_ttl_s() if session_ttl_s is None else session_ttl_s
        )
        self._ttl_scale = envs.VLLM_CPU_SESSION_TTL_K
        # Pin needs its own idle tracker unless a shared GPU tracker is attached.
        self._tracker = SessionLivenessTracker(observe_idle=True)
        self._first_hashes: dict[str, set[bytes]] = {}
        self._key_session: dict[OffloadKey, str] = {}

    def _remember_first(self, req_context: ReqContext) -> None:
        session_id = _session_id(req_context)
        if not session_id or _priority(req_context) != 0:
            return
        if self._first_hashes.get(session_id):
            return
        raw: set[bytes] = set()
        for item in req_context.block_hashes:
            if isinstance(item, (bytes, bytearray)):
                raw.add(bytes(item))
            else:
                try:
                    raw.add(bytes(item))
                except TypeError:
                    continue
        if raw or session_id not in self._first_hashes:
            self._first_hashes[session_id] = raw

    def _bind(self, key: OffloadKey, req_context: ReqContext) -> None:
        session_id = _session_id(req_context)
        if not session_id or _priority(req_context) != 0:
            return
        first = self._first_hashes.get(session_id)
        if first and get_offload_block_hash(key) in first:
            self._key_session[key] = session_id

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        keys = list(keys)
        super().touch(keys, req_context)
        for key in keys:
            self._bind(key, req_context)

    @override
    def note_stored(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        keys = list(keys)
        super().note_stored(keys, req_context)
        for key in keys:
            self._bind(key, req_context)

    @override
    def remove(self, key: OffloadKey) -> None:
        super().remove(key)
        self._key_session.pop(key, None)

    @override
    def on_new_request(self, req_context: ReqContext) -> None:
        self._remember_first(req_context)
        if self._shared_tracker is not None or self._tracker is None:
            return
        self._tracker.on_arrival(
            _session_id(req_context),
            self._time_fn(),
            priority=_priority(req_context),
            first_request_hashes=tuple(req_context.block_hashes),
        )

    def _held(self, key: OffloadKey, now: float) -> bool:
        session_id = self._key_session.get(key)
        if not session_id:
            return False
        tracker = self._active_tracker()
        if tracker is None:
            return False
        return not tracker.idle_expired(
            session_id, now, self._ttl_scale, self._ttl_floor
        )

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        now = self._time_fn()
        dead = self._dead_sessions()
        candidates = []
        for key in self._evictable - protected:
            # A pinned first-request block is held out of the peel until its
            # session goes idle. Everything else is scored by position tier,
            # so shallow prefixes go before the deep suffix (leaving a hole
            # the connector reloads as a disjoint span).
            if self._held(key, now):
                continue
            session_id = self._sessions.get(key)
            candidates.append(
                self._scorer.score(
                    self._positions.get(key, UNKNOWN_POSITION),
                    self._ranks.get(key, 0),
                    key,
                    session_id is not None and session_id in dead,
                )
            )
        if len(candidates) < n:
            return None
        evicted = []
        for *_, key in heapq.nsmallest(n, candidates):
            chunk = self.chunks.pop(key)
            self._forget(key)
            self._key_session.pop(key, None)
            evicted.append((key, chunk))
        return evicted

    @override
    def clear(self) -> None:
        super().clear()
        self._first_hashes.clear()
        self._key_session.clear()
