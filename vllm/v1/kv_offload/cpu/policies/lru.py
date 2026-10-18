# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class LRUCachePolicy(CachePolicy):
    """
    LRU Caching policy that keeps a dedicated evictable list for fast eviction.
    A use is indicated by,
     - First time the key is added (store).
     - Load job completion
     - touch
    """

    # The connector must not touch every key while the request is in flight.
    # That walk is reversed and makes the tail the eviction victim. Recency
    # is updated once, when the request finishes.
    refreshes_recency_from_connector = False

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        # Blocks with ref_cnt 0 (not participating in any loads/stores).
        # Eviction order is _rank, not this dict's order: a store that
        # completes late must not become newer than one inserted earlier.
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self._rank: dict[OffloadKey, int] = {}
        self._next_rank = 0

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def _bump(self, key: OffloadKey) -> None:
        self._rank[key] = self._next_rank
        self._next_rank += 1

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        self._bump(key)
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self._rank.pop(key, None)
        self.evictable_blocks.pop(key, None)

    @override
    def note_finished_keys(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        self.touch(keys, req_context)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        # Reverse so the first key, the head of the request, ends newest.
        for key in reversed(list(keys)):
            if key not in self._rank:
                continue
            self._bump(key)
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)

    @override
    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.blocks.clear()
        self._rank.clear()
        self._next_rank = 0

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        ranked = sorted(
            (key for key in self.evictable_blocks if key not in protected),
            key=self._rank.__getitem__,
        )
        if len(ranked) < n:
            return None
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key in ranked[:n]:
            block = self.blocks[key]
            assert block.ref_cnt == 0
            candidates.append((key, block))
            del self.evictable_blocks[key]
            del self.blocks[key]
            del self._rank[key]
        return candidates

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # blocks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self.evictable_blocks[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # key must have been in the evictable list.
        del self.evictable_blocks[key]
