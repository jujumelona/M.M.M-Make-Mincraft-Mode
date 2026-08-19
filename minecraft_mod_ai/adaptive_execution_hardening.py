from __future__ import annotations

from functools import wraps
from typing import Any

_EXPLORER_MARKER = "__mmm_capability_aware_explorer_v1__"
_SCALING_MARKER = "__mmm_budget_aware_test_time_scaling_v1__"


def harden_adaptive_execution() -> None:
    """Make optional retrieval/scaling depend on real runtime capabilities.

    Selective retrieval must not assume every router proxy exposes embedding or
    reranking, and automatic test-time scaling must not turn one native decode
    slot into duplicated serial work. Explicit operator opt-in remains able to
    spend additional sequential compute.
    """
    _harden_repository_explorer()
    _harden_test_time_scaling()


def _harden_repository_explorer() -> None:
    from .repository_explorer import RepositoryExplorer

    current = RepositoryExplorer.explore
    if getattr(current, _EXPLORER_MARKER, False):
        return

    @wraps(current)
    def explore(self: Any, query: str, *args: Any, **kwargs: Any):
        router = getattr(self, "router", None)
        if not _has_callable(router, "embed"):
            kwargs["semantic"] = False
        if not _has_callable(router, "rerank"):
            kwargs["rerank"] = False
        return current(self, query, *args, **kwargs)

    setattr(explore, _EXPLORER_MARKER, True)
    RepositoryExplorer.explore = explore


def _harden_test_time_scaling() -> None:
    from . import inference_time_scaling

    current = inference_time_scaling._scaling_mode
    if getattr(current, _SCALING_MARKER, False):
        return

    @wraps(current)
    def scaling_mode() -> str:
        mode = str(current())
        if mode != "auto":
            return mode
        try:
            from .max_efficiency_runtime_contract import _active_parallelism

            slots = int(_active_parallelism())
        except Exception:
            slots = 1
        return "auto" if slots > 1 else "off"

    setattr(scaling_mode, _SCALING_MARKER, True)
    inference_time_scaling._scaling_mode = scaling_mode


def _has_callable(owner: Any, name: str) -> bool:
    if owner is None:
        return False
    try:
        value = getattr(owner, name)
    except (AttributeError, RuntimeError, TypeError):
        return False
    return callable(value)


__all__ = ["harden_adaptive_execution"]
