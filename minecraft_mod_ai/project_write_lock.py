from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from .project_lock_state import ProjectLockState, project_lock_state


def _state_for(project_root: str | Path) -> ProjectLockState:
    return project_lock_state(project_root)


def _path_key(value: str | Path) -> str:
    rendered = str(value).strip().replace("\\", "/")
    path = PurePosixPath(rendered)
    if not rendered or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"project write path must be a safe relative path: {value!r}")
    return path.as_posix()


@contextmanager
def project_write_lock(
    project_root: str | Path,
    *,
    timeout_seconds: float | None = None,
) -> Iterator[None]:
    """Acquire the coarse project mutation gate.

    This is the compatibility boundary for read/merge/write operations whose exact
    target set is not known before reading shared project state. It excludes every
    path-scoped source transaction for the same project and remains re-entrant for
    existing higher-level generators. Waiting coarse writers receive preference over
    new scoped transactions so a stream of small patches cannot starve shared merges.
    """

    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool):
            raise ValueError("project write lock timeout must be a positive number")
        timeout_value = float(timeout_seconds)
        if timeout_value <= 0.0 or timeout_value != timeout_value:
            raise ValueError("project write lock timeout must be a positive finite number")
        deadline = time.monotonic() + timeout_value
    else:
        deadline = None

    state = _state_for(project_root)
    owner = threading.get_ident()
    with state.condition:
        if state.writer_owner == owner:
            state.writer_depth += 1
        else:
            if state.scoped_depth.get(owner, 0):
                raise RuntimeError(
                    "Cannot upgrade a path-scoped project write lock to the coarse "
                    "project lock; acquire the coarse lock first."
                )
            state.waiting_writers += 1
            state.condition.notify_all()
            try:
                while state.writer_owner is not None or state.scoped_total:
                    if deadline is None:
                        state.condition.wait()
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError(
                            "Timed out waiting for the coarse project write lock."
                        )
                    state.condition.wait(timeout=remaining)
                state.writer_owner = owner
                state.writer_depth = 1
            finally:
                state.waiting_writers -= 1
    try:
        yield
    finally:
        with state.condition:
            if state.writer_owner != owner or state.writer_depth < 1:
                raise RuntimeError("project write lock ownership was corrupted")
            state.writer_depth -= 1
            if state.writer_depth == 0:
                state.writer_owner = None
                state.condition.notify_all()


@contextmanager
def project_path_write_locks(
    project_root: str | Path,
    relative_paths: Iterable[str | Path],
) -> Iterator[None]:
    """Lock only the canonical target paths of one source transaction.

    Disjoint transactions may validate, stage, commit and roll back concurrently.
    Transactions sharing any target serialize on that target. The shared project gate
    prevents them from overlapping a coarse ``project_write_lock`` read/merge/write
    section. Path locks are acquired in lexical order to make multi-path transactions
    deadlock-free.
    """

    keys = tuple(sorted({_path_key(path) for path in relative_paths}))
    if not keys:
        raise ValueError("project path write lock requires at least one target path")
    state = _state_for(project_root)
    owner = threading.get_ident()

    with state.condition:
        owns_coarse = state.writer_owner == owner
        nested_scoped = state.scoped_depth.get(owner, 0) > 0
        if not owns_coarse:
            while state.writer_owner is not None or (
                state.waiting_writers and not nested_scoped
            ):
                state.condition.wait()
            state.scoped_total += 1
            state.scoped_depth[owner] = state.scoped_depth.get(owner, 0) + 1
        locks = tuple(state.path_locks.setdefault(key, threading.RLock()) for key in keys)

    acquired: list[threading.RLock] = []
    try:
        for lock in locks:
            lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()
        if not owns_coarse:
            with state.condition:
                depth = state.scoped_depth.get(owner, 0)
                if depth <= 1:
                    state.scoped_depth.pop(owner, None)
                else:
                    state.scoped_depth[owner] = depth - 1
                state.scoped_total -= 1
                if state.scoped_total < 0:
                    raise RuntimeError("project path write lock count was corrupted")
                if state.scoped_total == 0 or state.waiting_writers:
                    state.condition.notify_all()


__all__ = ["project_path_write_locks", "project_write_lock"]