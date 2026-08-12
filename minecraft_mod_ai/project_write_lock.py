from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_LOCKS_GUARD = threading.RLock()
_PROJECT_LOCKS: dict[str, threading.RLock] = {}


def _project_key(project_root: str | Path) -> str:
    return str(Path(project_root).expanduser().resolve())


def _lock_for(project_root: str | Path) -> threading.RLock:
    key = _project_key(project_root)
    with _LOCKS_GUARD:
        lock = _PROJECT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROJECT_LOCKS[key] = lock
        return lock


@contextmanager
def project_write_lock(project_root: str | Path) -> Iterator[None]:
    """Serialize project mutation while allowing re-entrant transactional helpers.

    Generation may do expensive planning/model work concurrently. Only the section
    that reads shared project metadata and commits source/resources should hold this
    lock. The lock is keyed by resolved project root, so independent projects never
    block each other.
    """

    lock = _lock_for(project_root)
    with lock:
        yield


__all__ = ["project_write_lock"]
