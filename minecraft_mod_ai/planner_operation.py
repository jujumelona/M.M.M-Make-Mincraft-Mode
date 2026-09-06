"""Request-local planner limits; ContextVars keep concurrent operations isolated."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_OUTPUT_LIMIT: ContextVar[int | None] = ContextVar("planner_output_limit", default=None)
_OPERATION: ContextVar[str] = ContextVar("planner_operation", default="")


def current_output_limit() -> int | None:
    return _OUTPUT_LIMIT.get()


def current_operation() -> str:
    return _OPERATION.get()


@contextmanager
def planner_operation(name: str, *, output_tokens: int) -> Iterator[None]:
    if output_tokens < 128:
        raise ValueError("A planner operation needs at least 128 output tokens")
    inherited = _OUTPUT_LIMIT.get()
    limit = min(inherited, output_tokens) if inherited is not None else output_tokens
    token = _OUTPUT_LIMIT.set(limit)
    operation = _OPERATION.set(name)
    try:
        yield
    finally:
        _OPERATION.reset(operation)
        _OUTPUT_LIMIT.reset(token)
