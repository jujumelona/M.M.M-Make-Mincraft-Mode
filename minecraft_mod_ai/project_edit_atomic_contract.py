from __future__ import annotations

from functools import wraps
from typing import Any

from .project_write_lock import project_write_lock


_MARKER = "_mmm_atomic_shared_project_edit"


def _wrap_shared_edit(function: Any):
    if getattr(function, _MARKER, False):
        return function

    @wraps(function)
    def atomic(info: Any, *args: Any, **kwargs: Any):
        # Keep expensive generation outside this lock.  These project_edit helpers
        # perform read/merge/SHA-checked writes to shared entrypoint/build metadata;
        # their complete read-modify-write transaction must be atomic across stages.
        with project_write_lock(info.root):
            return function(info, *args, **kwargs)

    setattr(atomic, _MARKER, True)
    return atomic


def install(project_edit_module: Any) -> None:
    for name in (
        "ensure_main_initializer_call",
        "ensure_client_entrypoint",
        "ensure_dependency",
    ):
        current = getattr(project_edit_module, name)
        setattr(project_edit_module, name, _wrap_shared_edit(current))


__all__ = ["install"]
