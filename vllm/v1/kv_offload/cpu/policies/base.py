# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from vllm.v1.kv_offload.base import OffloadKey, ReqContext


class BlockStatus(ctypes.Structure):
    """
    Offloading status for a single block of KV data.
    Holds the following information:

    ref_cnt - the current number of transfers using this block as a source.
        A value of -1 indicates the block is not yet ready to be read.
    block_id - index of the physical CPU buffer slot.
    """

    _fields_ = [("ref_cnt", ctypes.c_int32), ("block_id", ctypes.c_int64)]

    def __init__(self, block_id: int):
        super().__init__()
        # initialize block as "not ready" (ref_cnt = -1)
        self.ref_cnt = -1
        self.block_id = block_id

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the block is ready to be read.
        """
        return self.ref_cnt >= 0


class CachePolicy(ABC):
    """
    Encapsulates both block organization (data structures) and replacement
    decisions (which block to evict). LRU and ARC differ in both dimensions —
    ARC's ghost lists and target_t1_size live at the intersection of storage
    and eviction, so they cannot be separated cleanly.

    ``supports_disjoint_spans`` stays false for LRU and ARC. Lookup then
    remains one contiguous run. Prefix and first-request policies set it
    so a later cached run can be loaded after the hole in front of it is
    recomputed.
    """

    supports_disjoint_spans: bool = False

    # The connector refreshes recency on lookup and store when this is set.
    # LRU records recency at insertion and when a store completes, so a
    # refresh of every key inverts that order and evicts the tail.
    refreshes_recency_from_connector: bool = True

    @abstractmethod
    def __init__(self, cache_capacity: int) -> None: ...

    @abstractmethod
    def get(self, key: OffloadKey) -> BlockStatus | None:
        """Find block in data structures. Returns None if not present."""

    @abstractmethod
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        """Add a newly allocated block. For ARC: also removes from ghost lists."""

    @abstractmethod
    def remove(self, key: OffloadKey) -> None:
        """Remove a block (used to clean up after a failed store)."""

    def note_finished_keys(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        """Refresh recency once, after the request has finished storing.

        The default is a no-op. LRU and ARC override this. A per-step touch
        is a different event: for ARC it means a reuse, not a store.
        """
        return

    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        """
        Mark blocks as recently used.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context for the request touching these blocks.
        """

    @abstractmethod
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """
        Evict exactly n blocks, skipping any in protected.

        Returns a list of (key, block) for the evicted blocks,
        or None if n evictions cannot be satisfied. The operation is atomic:
        if None is returned, no state changes are made.

        For ARC: ghost list cleanup (trimming to cache_capacity) is performed
        at the end of a successful eviction.
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Remove ALL blocks regardless of ref_cnt.

        Ghost lists and adaptive state are also reset.
        """

    def cached_spans(
        self, keys: Sequence[OffloadKey]
    ) -> list[tuple[int, int]] | None:
        """Cached runs as ``[start, end)`` indexes into ``keys``.

        ``None`` means this policy does not describe holes. The connector
        keeps the contiguous prefix scan. An empty list means nothing in
        ``keys`` is cached.
        """
        if not self.supports_disjoint_spans:
            return None
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for index, key in enumerate(keys):
            block = self.get(key)
            cached = block is not None and block.ref_cnt >= 0
            if cached and start is None:
                start = index
            elif not cached and start is not None:
                spans.append((start, index))
                start = None
        if start is not None:
            spans.append((start, len(keys)))
        return spans

    def on_new_request(self, req_context: ReqContext) -> None:
        return

    def on_request_finished(self, req_context: ReqContext) -> None:
        return

    def note_stored(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        """Called after a CPU store is allocated. Default does nothing."""
        return

    def mark_evictable(self, key: OffloadKey) -> None:
        """Called when a block's ref_cnt transitions to 0."""
        return

    def mark_non_evictable(self, key: OffloadKey) -> None:
        """Called when a block's ref_cnt transitions from 0."""
        return
