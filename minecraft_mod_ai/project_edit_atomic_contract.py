from __future__ import annotations

from functools import wraps
from typing import Any, Iterable

from .project_write_lock import project_write_lock


_MARKER = "_mmm_atomic_shared_project_edit"
_SHARED_EDIT_NAMES = (
    "ensure_main_initializer_call",
    "ensure_client_entrypoint",
    "ensure_dependency",
)


def _wrap_shared_edit(function: Any):
    if getattr(function, _MARKER, False):
        return function

    @wraps(function)
    def atomic(info: Any, *args: Any, **kwargs: Any):
        # Keep expensive generation outside this lock. These project_edit helpers
        # perform read/merge/SHA-checked writes to shared entrypoint/build metadata;
        # their complete read-modify-write transaction must be atomic across stages.
        with project_write_lock(info.root):
            return function(info, *args, **kwargs)

    setattr(atomic, _MARKER, True)
    return atomic


def install(
    project_edit_module: Any,
    *,
    consumers: Iterable[Any] = (),
) -> None:
    originals: dict[str, Any] = {}
    for name in _SHARED_EDIT_NAMES:
        current = getattr(project_edit_module, name)
        wrapped = _wrap_shared_edit(current)
        setattr(project_edit_module, name, wrapped)
        originals[name] = wrapped

    # Some generator modules import helpers directly (``from project_edit import``)
    # before runtime contracts are installed. Rebind those cached references so the
    # atomic boundary is not bypassed merely because of import order.
    for consumer in consumers:
        for name, wrapped in originals.items():
            if hasattr(consumer, name):
                setattr(consumer, name, wrapped)


__all__ = ["install"]
