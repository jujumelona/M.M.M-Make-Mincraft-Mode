from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar


_T = TypeVar("_T")
_MODEL_EXECUTION_DEADLINE: ContextVar[float | None] = ContextVar(
    "mmm_model_execution_deadline",
    default=None,
)
_DEFAULT_PLANNING_WORK_UNIT_TIMEOUT_SECONDS = 600.0


class ModelExecutionDeadlineExceeded(TimeoutError):
    """The current model-backed work unit exhausted its monotonic execution budget."""


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or value <= 0.0:
        return default
    return value


def planning_work_unit_timeout_seconds() -> float:
    """Return the bounded wall-clock budget for one planning/research work unit.

    This is an execution-safety deadline, not a semantic retry/count cap. Large plans
    receive proportionally larger stage budgets based on their actual number of work
    units; a single blocked unit still cannot pin the scheduler indefinitely.
    """

    return _positive_float_env(
        "MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS",
        _DEFAULT_PLANNING_WORK_UNIT_TIMEOUT_SECONDS,
    )


def planning_stage_deadline(
    *,
    work_units: int,
    workers: int,
    started_at: float | None = None,
) -> float:
    """Return a monotonic stage deadline scaled to the required parallel work waves."""

    started = time.monotonic() if started_at is None else float(started_at)
    unit_count = max(1, int(work_units))
    worker_count = max(1, int(workers))
    waves = math.ceil(unit_count / worker_count)
    return started + planning_work_unit_timeout_seconds() * waves


def current_model_execution_deadline() -> float | None:
    return _MODEL_EXECUTION_DEADLINE.get()


def remaining_model_execution_seconds(
    deadline_monotonic: float | None = None,
) -> float | None:
    deadline = (
        current_model_execution_deadline()
        if deadline_monotonic is None
        else float(deadline_monotonic)
    )
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


@contextmanager
def bind_model_execution_deadline(deadline_monotonic: float) -> Iterator[float]:
    """Bind the earliest execution deadline to the current context."""

    requested = float(deadline_monotonic)
    existing = current_model_execution_deadline()
    effective = requested if existing is None else min(existing, requested)
    token = _MODEL_EXECUTION_DEADLINE.set(effective)
    try:
        yield effective
    finally:
        _MODEL_EXECUTION_DEADLINE.reset(token)


def run_with_model_execution_deadline(
    deadline_monotonic: float,
    function: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Run a callable with a deadline that propagates through copied contexts."""

    with bind_model_execution_deadline(deadline_monotonic):
        return function(*args, **kwargs)


def _condition_wait(condition: threading.Condition) -> None:
    remaining = remaining_model_execution_seconds()
    if remaining is not None:
        if remaining <= 0.0:
            raise ModelExecutionDeadlineExceeded(
                "model execution deadline expired while waiting for shared model capacity"
            )
        condition.wait(timeout=remaining)
        if remaining_model_execution_seconds() == 0.0:
            raise ModelExecutionDeadlineExceeded(
                "model execution deadline expired while waiting for shared model capacity"
            )
        return
    condition.wait()


class ReentrantReadWriteLock:
    """Writer-reentrant lock with shared readers and writer preference."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._writer_thread: int | None = None
        self._writer_depth = 0
        self._readers: dict[int, int] = {}
        self._suspended_reads: dict[int, int] = {}
        self._waiting_writers = 0

    def acquire(self) -> bool:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_thread == owner:
                self._writer_depth += 1
                return True

            # A shared owner may need exclusive GPU access while still inside the
            # outer shared scope (for example, a tool-backed generation path that
            # invokes another exclusive model). Suspend that owner's read claim
            # while it waits so two concurrent read->write upgrades cannot pin each
            # other forever. The read depth is restored when the exclusive section
            # exits, preserving the enclosing shared scope.
            suspended = self._readers.pop(owner, 0)
            if suspended:
                self._condition.notify_all()

            self._waiting_writers += 1
            try:
                while self._writer_thread is not None or self._readers:
                    _condition_wait(self._condition)
                self._writer_thread = owner
                self._writer_depth = 1
                if suspended:
                    self._suspended_reads[owner] = (
                        self._suspended_reads.get(owner, 0) + suspended
                    )
                return True
            except BaseException:
                if suspended:
                    self._readers[owner] = self._readers.get(owner, 0) + suspended
                    self._condition.notify_all()
                raise
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
                suspended = self._suspended_reads.pop(owner, 0)
                if suspended:
                    self._readers[owner] = self._readers.get(owner, 0) + suspended
                self._condition.notify_all()

    def acquire_read(self) -> bool:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_thread == owner:
                self._readers[owner] = self._readers.get(owner, 0) + 1
                return True
            while self._writer_thread is not None or self._waiting_writers > 0:
                _condition_wait(self._condition)
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
                _condition_wait(self._condition)
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
    """Return the validated active llama slot count without imposing a second cap."""

    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def router_owns_native_model(router: Any, *, role: str = "planner") -> bool:
    """Return True only when this router proves ownership of the local native model."""

    try:
        config = router.registry.role(router.profile, role)
    except Exception:
        return False
    return (
        bool(getattr(config, "exclusive_gpu", False))
        and str(getattr(config, "provider", "")) == "local"
        and str(getattr(config, "adapter", "")) in {"llama_cpp", "vllm"}
    )


def router_native_model_parallelism(router: Any, *, role: str = "planner") -> int:
    """Return measured native-model slots, or one for unproven/remote routers."""

    if not router_owns_native_model(router, role=role):
        return 1
    return active_llama_parallelism()


__all__ = [
    "ModelExecutionDeadlineExceeded",
    "ReentrantCapacityGate",
    "ReentrantReadWriteLock",
    "active_llama_parallelism",
    "bind_model_execution_deadline",
    "current_model_execution_deadline",
    "planning_stage_deadline",
    "planning_work_unit_timeout_seconds",
    "remaining_model_execution_seconds",
    "router_native_model_parallelism",
    "router_owns_native_model",
    "run_with_model_execution_deadline",
]