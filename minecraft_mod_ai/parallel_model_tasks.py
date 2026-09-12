from __future__ import annotations

"""Deterministic host-side scheduling for independent small-model calls."""

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, TypeVar

from .model_concurrency import router_native_model_parallelism

_T = TypeVar("_T")
_R = TypeVar("_R")


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

    Remote/unproven routers stay serial. ContextVars are copied into each worker, output
    order is deterministic, and any failure cancels work that has not started yet.
    """

    values = tuple(items)
    if not values:
        return []
    workers = max(
        1,
        min(len(values), router_native_model_parallelism(router, role=role)),
    )
    if workers == 1:
        return [worker(item) for item in values]

    contexts = [copy_context() for _ in values]
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=thread_name_prefix,
    ) as pool:
        futures = [
            pool.submit(contexts[index].run, worker, value)
            for index, value in enumerate(values)
        ]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


__all__ = ["deterministic_model_map", "serialized_callback"]
