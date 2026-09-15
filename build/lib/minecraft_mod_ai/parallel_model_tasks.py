from __future__ import annotations

"""Deterministic host-side scheduling for independent small-model calls."""

import threading
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any, TypeVar

from .deadline_executor import iter_completed_with_deadlines
from .model_concurrency import router_native_model_parallelism

_T = TypeVar("_T")
_R = TypeVar("_R")
_MODEL_PARALLEL_DEPTH: ContextVar[int] = ContextVar("mmm_model_parallel_depth", default=0)


def serialized_callback(callback: Callable[..., _R] | None) -> Callable[..., _R] | None:
    """Serialize a callback shared by parallel workers without serializing model work."""

    if callback is None:
        return None
    lock = threading.RLock()

    def invoke(*args: Any, **kwargs: Any) -> _R:
        with lock:
            return callback(*args, **kwargs)

    return invoke


def deterministic_model_map(
    router: Any,
    items: Iterable[_T],
    worker: Callable[[_T], _R],
    *,
    role: str = "planner",
    thread_name_prefix: str = "mmm-model-task",
) -> list[_R]:
    """Run independent calls on measured native slots and return input-order results.

    Remote/unproven routers stay serial. Parallel work is scheduled through the shared
    deadline executor so one blocked provider call cannot pin the caller indefinitely.
    Nested maps stay serial because the outer map already owns the measured model width.
    """

    values = tuple(items)
    if not values:
        return []

    nested = _MODEL_PARALLEL_DEPTH.get() > 0
    workers = 1 if nested else max(
        1,
        min(len(values), router_native_model_parallelism(router, role=role)),
    )
    if workers == 1:
        token = _MODEL_PARALLEL_DEPTH.set(_MODEL_PARALLEL_DEPTH.get() + 1)
        try:
            return [worker(item) for item in values]
        finally:
            _MODEL_PARALLEL_DEPTH.reset(token)

    def run_indexed(item: tuple[int, _T]) -> tuple[int, _R]:
        index, value = item
        token = _MODEL_PARALLEL_DEPTH.set(_MODEL_PARALLEL_DEPTH.get() + 1)
        try:
            return index, worker(value)
        finally:
            _MODEL_PARALLEL_DEPTH.reset(token)

    indexed = tuple(enumerate(values))
    results: dict[int, _R] = {}
    for _item, indexed_result in iter_completed_with_deadlines(
        indexed,
        run_indexed,
        max_workers=workers,
        stage=thread_name_prefix,
        sort_key=lambda item: item[0],
    ):
        index, result = indexed_result
        results[index] = result

    if len(results) != len(values):
        raise RuntimeError(f"{thread_name_prefix}: parallel map lost a completed result")
    return [results[index] for index in range(len(values))]


__all__ = ["deterministic_model_map", "serialized_callback"]
