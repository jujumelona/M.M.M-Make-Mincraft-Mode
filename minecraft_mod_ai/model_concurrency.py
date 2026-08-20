from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Callable, Iterator


class ReentrantReadWriteLock:
    """Writer-reentrant lock with shared readers and writer preference."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._writer_thread: int | None = None
        self._writer_depth = 0
        self._readers: dict[int, int] = {}
        self._waiting_writers = 0

    def acquire(self) -> bool:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_thread == owner:
                self._writer_depth += 1
                return True
            self._waiting_writers += 1
            try:
                while self._writer_thread is not None or self._other_reader_count(owner):
                    self._condition.wait()
                self._writer_thread = owner
                self._writer_depth = 1
                return True
            finally:
                self._waiting_writers -= 1

    def release(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_thread != owner or self._writer_depth <= 0:
                raise RuntimeError("cannot release unowned GPU write lock")
            self._writer_depth -= 1
            if self._writer_depth == 0:
                self._writer_thread = None
                self._condition.notify_all()

    def acquire_read(self) -> bool:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_thread == owner:
                self._readers[owner] = self._readers.get(owner, 0) + 1
                return True
            while self._writer_thread is not None or self._waiting_writers > 0:
                self._condition.wait()
            self._readers[owner] = self._readers.get(owner, 0) + 1
            return True

    def release_read(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            count = self._readers.get(owner, 0)
            if count <= 0:
                raise RuntimeError("cannot release unowned GPU read lock")
            if count == 1:
                self._readers.pop(owner, None)
            else:
                self._readers[owner] = count - 1
            if not self._readers:
                self._condition.notify_all()

    def _other_reader_count(self, owner: int) -> int:
        return sum(
            count
            for thread_id, count in self._readers.items()
            if thread_id != owner
        )

    def __enter__(self) -> ReentrantReadWriteLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    @contextmanager
    def shared(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()


class ReentrantCapacityGate:
    """Bound concurrent callers to a dynamic capacity without blocking re-entry."""

    def __init__(self, capacity: Callable[[], int]) -> None:
        self._capacity = capacity
        self._condition = threading.Condition(threading.RLock())
        self._owners: dict[int, int] = {}

    def acquire(self) -> bool:
        owner = threading.get_ident()
        with self._condition:
            depth = self._owners.get(owner, 0)
            if depth:
                self._owners[owner] = depth + 1
                return True
            while len(self._owners) >= max(1, int(self._capacity())):
                self._condition.wait()
            self._owners[owner] = 1
            return True

    def release(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            depth = self._owners.get(owner, 0)
            if depth <= 0:
                raise RuntimeError("cannot release unowned llama inference slot")
            if depth == 1:
                self._owners.pop(owner, None)
                self._condition.notify_all()
            else:
                self._owners[owner] = depth - 1

    def __enter__(self) -> ReentrantCapacityGate:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


def active_llama_parallelism() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


__all__ = [
    "ReentrantCapacityGate",
    "ReentrantReadWriteLock",
    "active_llama_parallelism",
]
