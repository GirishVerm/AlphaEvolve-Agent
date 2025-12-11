import time
import threading
from typing import Any, Optional


class ThreadSafeLRUCacheTTL:
    """
    Thread-safe LRU Cache with TTL expiration and O(1) get/put operations.

    Features:
    - get(key) -> value or None if missing/expired
    - put(key, value) -> store/update value with TTL
    - TTL expiration (per-cache TTL, same for all entries; if ttl is None, no expiration)
    - LRU eviction when capacity is reached
    - Uses a re-entrant lock for thread-safety
    - O(1) get/put via a hashmap + doubly linked list
    """

    class _Node:
        __slots__ = ("key", "value", "prev", "next", "expiry")

        def __init__(self, key: Any, value: Any, expiry: Optional[float]):
            self.key = key
            self.value = value
            self.prev: Optional["ThreadSafeLRUCacheTTL._Node"] = None
            self.next: Optional["ThreadSafeLRUCacheTTL._Node"] = None
            self.expiry: Optional[float] = expiry

    def __init__(self, capacity: int, ttl: Optional[float] = None):
        if capacity <= 0:
            raise ValueError("Capacity must be > 0")
        self.capacity = capacity
        self.ttl = ttl  # in seconds; None means no expiration
        self.lock = threading.RLock()

        # dictionary for O(1) access: key -> Node
        self.cache: dict[Any, ThreadSafeLRUCacheTTL._Node] = {}

        # Dummy head and tail to simplify add/remove
        self.head = self._Node(None, None, None)
        self.tail = self._Node(None, None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    # Internal helpers

    def _is_expired(self, node: "ThreadSafeLRUCacheTTL._Node") -> bool:
        if node.expiry is None:
            return False
        return time.time() > node.expiry

    def _add_to_head(self, node: "ThreadSafeLRUCacheTTL._Node") -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: "ThreadSafeLRUCacheTTL._Node") -> None:
        prev = node.prev
        nxt = node.next
        if prev:
            prev.next = nxt
        if nxt:
            nxt.prev = prev
        node.prev = None
        node.next = None

    def _move_to_head(self, node: "ThreadSafeLRUCacheTTL._Node") -> None:
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional["ThreadSafeLRUCacheTTL._Node"]:
        # Remove and return the least-recently-used real node
        node = self.tail.prev
        if node is self.head:
            return None
        self._remove_node(node)
        return node

    def _evict_expired_from_tail(self) -> None:
        # Clean up any expired nodes starting from the tail (LRU side)
        while True:
            if self.tail.prev is self.head:
                break
            tail_node = self.tail.prev
            if self._is_expired(tail_node):
                self._remove_node(tail_node)
                del self.cache[tail_node.key]
            else:
                break

    # Public API

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value for key if present and not expired.
        Returns None if key is missing or expired.
        """
        with self.lock:
            node = self.cache.get(key, None)
            if node is None:
                return None

            if self._is_expired(node):
                # Remove expired node
                self._remove_node(node)
                del self.cache[key]
                return None

            # Move accessed node to head (most recently used)
            self._move_to_head(node)
            return node.value

    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update value for key with TTL handling.
        If capacity is reached, evict the LRU item (after cleaning expired ones).
        """
        with self.lock:
            # If TTL is not enabled, expiry remains None
            expiry = None if self.ttl is None else time.time() + self.ttl

            if key in self.cache:
                node = self.cache[key]
                node.value = value
                node.expiry = expiry
                self._move_to_head(node)
                return

            # Cleanup expired items from the tail first
            self._evict_expired_from_tail()

            # Evict LRU if at capacity
            if len(self.cache) >= self.capacity:
                tail = self._pop_tail()
                if tail is not None:
                    del self.cache[tail.key]

            # Create new node and insert at head
            new_node = self._Node(key, value, expiry)
            self.cache[key] = new_node
            self._add_to_head(new_node)