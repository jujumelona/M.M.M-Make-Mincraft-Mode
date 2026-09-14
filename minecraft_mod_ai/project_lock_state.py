from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectLockState:
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )
    writer_owner: int | None = None
    writer_depth: int = 0
    waiting_writers: int = 0
    scoped_total: int = 0
    scoped_depth: dict[int, int] = field(default_factory=dict)
    path_locks: dict[str, threading.RLock] = field(default_factory=dict)


_STATES: dict[Path, ProjectLockState] = {}
_STATES_LOCK = threading.Lock()


def project_lock_state(project_root: str | Path) -> ProjectLockState:
    root = Path(project_root).expanduser().resolve()
    with _STATES_LOCK:
        state = _STATES.get(root)
        if state is None:
            state = ProjectLockState()
            _STATES[root] = state
        return state


__all__ = ["ProjectLockState", "project_lock_state"]
