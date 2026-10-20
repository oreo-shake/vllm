# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatic system-prompt pinning for the v1 prefix cache.

Prefix-first eviction peels leading blocks first, which is exactly the
shared system prompt. Automatic Prefix Caching will reuse that prefix
only while it stays resident; under pressure it is the first thing we
throw away. Pinning holds those blocks with an extra ref so they never
enter the free queue.

Detection is *not* a hardcoded token length. It follows the coverage-
cliff idea (and the reusable-module view of Prompt Cache / RadixAttention):

1. Watch the first turn of each unique session (or request_id if the
   request has no session). Later turns of the same chat would stretch
   the shared prefix into conversation history — that is not a system
   prompt.
2. Cluster those first-turns by the first block hash so two tenants with
   different system prompts do not cancel each other.
3. For each cluster, walk from the left and build a monotone coverage
   curve: at position i, the fraction of observations whose prefix
   ``[0, i]`` equals the running mode prefix. The curve is
   non-increasing.
4. The system-prompt boundary is the last index that stays at or above
   ``share`` before a drop of at least ``drop``, or before coverage
   falls below ``share``. Cap at ``max_pin_blocks`` so a shared document
   after the prompt cannot pin the whole cache.
5. Require ``min_requests`` unique first-turns in the cluster, then the
   same candidate on ``confirm`` consecutive detections, before the pin
   is committed. If later first-turns stop carrying that prefix
   (coverage < ``unpin_share``), unpin and look again.

References:
* Gim et al., "Prompt Cache: Modular Attention Reuse for Low-Latency
  Inference", MLSys 2024 (arXiv:2311.04934) — system prompts as
  reusable KV modules that should persist on the server.
* Zheng et al., "SGLang / RadixAttention" (arXiv:2312.07104) — leaf-only
  LRU so shared prefix ancestors stay resident.
* vLLM Automatic Prefix Caching — hash-chained blocks; this module only
  adds an eviction exemption for the detected leading hashes.
* OpenAI prompt caching / LMCache ``pin(tokens)`` / TensorRT-LLM
  ``save_system_prompt_kv_cache`` — explicit pin of a known prefix.
  We infer that prefix from traffic instead of taking a string.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Hashable

PrefixKey = Hashable


def coverage_curve(
    observations: list[tuple[PrefixKey, ...]],
    max_blocks: int,
) -> tuple[tuple[PrefixKey, ...], tuple[float, ...]]:
    """Mode prefix and per-position coverage of ``observations``.

    ``coverage[i]`` is ``count(obs[: i + 1] == mode[: i + 1]) / n``.
    The sequence is monotonically non-increasing.
    """
    n = len(observations)
    if n == 0 or max_blocks <= 0:
        return (), ()
    mode: list[PrefixKey] = []
    coverage: list[float] = []
    active = list(observations)
    limit = min(max_blocks, max(len(obs) for obs in observations))
    for i in range(limit):
        remaining = [obs for obs in active if len(obs) > i]
        if not remaining:
            break
        hash_i, count_i = Counter(obs[i] for obs in remaining).most_common(1)[0]
        cov = count_i / n
        mode.append(hash_i)
        coverage.append(cov)
        active = [obs for obs in remaining if obs[i] == hash_i]
    return tuple(mode), tuple(coverage)


def cliff_length(
    coverage: tuple[float, ...],
    share: float,
    drop: float,
) -> int:
    """Longest leading run that stays on the high plateau.

    Includes every position with coverage >= ``share`` until the next
    step drops by at least ``drop`` or falls below ``share``.
    """
    if not coverage or coverage[0] < share:
        return 0
    length = 0
    for i, cov in enumerate(coverage):
        if cov < share:
            break
        length = i + 1
        nxt = coverage[i + 1] if i + 1 < len(coverage) else 0.0
        if cov - nxt >= drop or nxt < share:
            break
    return length


def prefix_share(
    prefix: tuple[PrefixKey, ...],
    observations: list[tuple[PrefixKey, ...]],
) -> float:
    if not prefix or not observations:
        return 0.0
    hits = 0
    n = len(prefix)
    for obs in observations:
        if len(obs) >= n and obs[:n] == prefix:
            hits += 1
    return hits / len(observations)


@dataclass
class PinSnapshot:
    """Result of one observe() that changed the committed pin set."""

    hashes: tuple[PrefixKey, ...]
    clusters: int
    n_obs: int
    coverage: tuple[float, ...]
    cliff: int


@dataclass
class SystemPromptPinDetector:
    """Online detector for a carried-forward system-prompt prefix."""

    min_requests: int = 8
    share: float = 0.75
    drop: float = 0.20
    max_pin_blocks: int = 512
    max_probe_blocks: int = 1024
    confirm: int = 2
    window: int = 64
    max_clusters: int = 4
    unpin_share: float | None = None

    _seen: set[str] = field(default_factory=set, init=False, repr=False)
    _order: deque[str] = field(default_factory=deque, init=False, repr=False)
    _obs: dict[str, tuple[PrefixKey, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _candidate: dict[PrefixKey, tuple[PrefixKey, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _confirm_n: dict[PrefixKey, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _pinned: dict[PrefixKey, tuple[PrefixKey, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_hashes: tuple[PrefixKey, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_requests < 2:
            raise ValueError(f"min_requests must be >= 2, got {self.min_requests}")
        if not 0.0 < self.share <= 1.0:
            raise ValueError(f"share must be in (0, 1], got {self.share}")
        if not 0.0 < self.drop <= 1.0:
            raise ValueError(f"drop must be in (0, 1], got {self.drop}")
        if self.max_pin_blocks < 1:
            raise ValueError(f"max_pin_blocks must be >= 1, got {self.max_pin_blocks}")
        if self.max_probe_blocks < 1:
            raise ValueError(
                f"max_probe_blocks must be >= 1, got {self.max_probe_blocks}"
            )
        if self.confirm < 1:
            raise ValueError(f"confirm must be >= 1, got {self.confirm}")
        if self.window < self.min_requests:
            raise ValueError(
                f"window ({self.window}) must be >= min_requests ({self.min_requests})"
            )
        if self.max_clusters < 1:
            raise ValueError(f"max_clusters must be >= 1, got {self.max_clusters}")
        if self.unpin_share is None:
            self.unpin_share = self.share * 0.5
        elif not 0.0 <= self.unpin_share <= self.share:
            raise ValueError(
                f"unpin_share must be in [0, share], got {self.unpin_share}"
            )

    @property
    def pinned_hashes(self) -> tuple[PrefixKey, ...]:
        return tuple(h for prefix in self._pinned.values() for h in prefix)

    @property
    def n_unique(self) -> int:
        return len(self._obs)

    def observe(
        self,
        session_id: str,
        hashes: tuple[PrefixKey, ...],
    ) -> PinSnapshot | None:
        """Record one unique first-turn. Return a snapshot when the pin set changes."""
        if not session_id or not hashes:
            return None
        if session_id in self._seen:
            return None
        self._seen.add(session_id)
        prefix = tuple(hashes[: self.max_probe_blocks])
        self._order.append(session_id)
        self._obs[session_id] = prefix
        while len(self._order) > self.window:
            old = self._order.popleft()
            self._obs.pop(old, None)

        clusters: dict[PrefixKey, list[tuple[PrefixKey, ...]]] = defaultdict(list)
        for obs in self._obs.values():
            if obs:
                clusters[obs[0]].append(obs)

        ranked = sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)
        next_pinned: dict[PrefixKey, tuple[PrefixKey, ...]] = {}
        best_coverage: tuple[float, ...] = ()
        best_cliff = 0
        for key, obs_list in ranked:
            if len(next_pinned) >= self.max_clusters:
                break
            decided = self._decide_cluster(key, obs_list)
            if decided is None:
                continue
            mode, coverage, cliff = decided
            next_pinned[key] = mode
            if cliff > best_cliff:
                best_coverage = coverage
                best_cliff = cliff

        new_hashes = tuple(h for prefix in next_pinned.values() for h in prefix)
        self._pinned = next_pinned
        if new_hashes == self._last_hashes:
            return None
        self._last_hashes = new_hashes
        return PinSnapshot(
            hashes=new_hashes,
            clusters=len(next_pinned),
            n_obs=len(self._obs),
            coverage=best_coverage[:best_cliff],
            cliff=best_cliff,
        )

    def _decide_cluster(
        self,
        key: PrefixKey,
        obs_list: list[tuple[PrefixKey, ...]],
    ) -> tuple[tuple[PrefixKey, ...], tuple[float, ...], int] | None:
        current = self._pinned.get(key)
        unpin_share = self.unpin_share if self.unpin_share is not None else 0.0
        if current is not None and prefix_share(current, obs_list) < unpin_share:
            self._candidate.pop(key, None)
            self._confirm_n.pop(key, None)
            return None

        if len(obs_list) < self.min_requests:
            if current is not None:
                return current, (), len(current)
            return None

        mode, coverage = coverage_curve(obs_list, self.max_pin_blocks)
        cliff = cliff_length(coverage, self.share, self.drop)
        candidate = mode[:cliff]
        if not candidate:
            return (current, (), len(current)) if current is not None else None

        if current == candidate:
            self._candidate[key] = candidate
            self._confirm_n[key] = self.confirm
            return candidate, coverage, cliff

        prev = self._candidate.get(key)
        if prev == candidate:
            self._confirm_n[key] = self._confirm_n.get(key, 0) + 1
        else:
            self._candidate[key] = candidate
            self._confirm_n[key] = 1

        if self._confirm_n[key] >= self.confirm:
            return candidate, coverage, cliff
        if current is not None:
            return current, (), len(current)
        return None
