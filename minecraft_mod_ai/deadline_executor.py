from __future__ import annotations

"""Deadline-aware parallel execution for model-backed planning work.

Callers retain semantic ownership of retries, checkpoints, and failure policy. This
module owns bounded scheduling, monotonic work/stage deadlines, context propagation,
cancellation, and executor shutdown. Streaming callers receive each completed result
as soon as it is ready so executor-side result retention stays bounded by active model
parallelism instead of growing with the full planning workload.
"""

from collections.abc import Callable, Iterable, Iterator, Sized
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
import time
from typing import Generic, TypeVar

from .model_concurrency import (
    planning_stage_deadline,
    planning_work_unit_timeout_seconds,
    run_with_model_execution_deadline,
)


_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class _ActiveTask(Generic[_Item]):
    sequence: int
    item: _Item
    deadline: float
    submitted_at: float


@dataclass
class _SubmissionState(Generic[_Item]):
    source: Iterator[_Item]
    sequence: int = 0
    exhausted: bool = False


class ParallelExecutionTimeout(TimeoutError):
    """A bounded parallel stage or one of its atomic work units exceeded its deadline."""

    def __init__(
        self,
        *,
        stage: str,
        item: object,
        elapsed_seconds: float,
        work_unit_timeout_seconds: float,
        deadline_kind: str,
    ) -> None:
        self.stage = str(stage)
        self.item = item
        self.elapsed_seconds = max(0.0, float(elapsed_seconds))
        self.work_unit_timeout_seconds = float(work_unit_timeout_seconds)
        self.deadline_kind = str(deadline_kind)
        super().__init__(
            f"{self.stage} {self.deadline_kind} deadline exceeded after "
            f"{self.elapsed_seconds:.3f}s; work_unit_timeout_seconds="
            f"{self.work_unit_timeout_seconds:.3f}; item={self.item!r}"
        )


class ParallelTaskError(RuntimeError):
    """Preserve the work-item identity when a completed future raises."""

    def __init__(self, *, stage: str, item: object, cause: BaseException) -> None:
        self.stage = str(stage)
        self.item = item
        self.cause = cause
        super().__init__(f"{self.stage} worker failed for {self.item!r}: {cause}")


def _resolve_worker_count(total_units: int | None, max_workers: int) -> int:
    workers = max(1, int(max_workers))
    return min(workers, total_units) if total_units is not None else workers


def _submit_next(
    state: _SubmissionState[_Item],
    active: dict[Future[_Result], _ActiveTask[_Item]],
    pool: ThreadPoolExecutor,
    worker: Callable[[_Item], _Result],
    *,
    unit_timeout: float,
    stage_deadline: float | None,
) -> bool:
    if state.exhausted:
        return False
    try:
        item = next(state.source)
    except StopIteration:
        state.exhausted = True
        return False

    submitted_at = time.monotonic()
    deadline = submitted_at + unit_timeout
    if stage_deadline is not None:
        deadline = min(stage_deadline, deadline)
    context = copy_context()
    future = pool.submit(
        context.run,
        run_with_model_execution_deadline,
        deadline,
        worker,
        item,
    )
    active[future] = _ActiveTask(
        sequence=state.sequence,
        item=item,
        deadline=deadline,
        submitted_at=submitted_at,
    )
    state.sequence += 1
    return True


def _fill_active_slots(
    state: _SubmissionState[_Item],
    active: dict[Future[_Result], _ActiveTask[_Item]],
    pool: ThreadPoolExecutor,
    worker: Callable[[_Item], _Result],
    *,
    workers: int,
    unit_timeout: float,
    stage_deadline: float | None,
) -> None:
    while len(active) < workers and _submit_next(
        state,
        active,
        pool,
        worker,
        unit_timeout=unit_timeout,
        stage_deadline=stage_deadline,
    ):
        pass


def _next_completed_batch(
    active: dict[Future[_Result], _ActiveTask[_Item]],
    *,
    stage: str,
    unit_timeout: float,
    stage_deadline: float | None,
) -> set[Future[_Result]]:
    now = time.monotonic()
    wake_deadline = min(meta.deadline for meta in active.values())
    if stage_deadline is not None:
        wake_deadline = min(stage_deadline, wake_deadline)
    done, _ = wait(
        tuple(active),
        timeout=max(0.0, wake_deadline - now),
        return_when=FIRST_COMPLETED,
    )
    if done:
        return done

    now = time.monotonic()
    deadline_kind = (
        "stage"
        if stage_deadline is not None and now >= stage_deadline
        else "work_unit"
    )
    expired = min(active.values(), key=lambda meta: (meta.deadline, meta.sequence))
    raise ParallelExecutionTimeout(
        stage=stage,
        item=expired.item,
        elapsed_seconds=now - expired.submitted_at,
        work_unit_timeout_seconds=unit_timeout,
        deadline_kind=deadline_kind,
    )


def _completion_order(
    future: Future[_Result],
    active: dict[Future[_Result], _ActiveTask[_Item]],
    sort_key: Callable[[_Item], object] | None,
) -> tuple[object, int]:
    meta = active[future]
    key = sort_key(meta.item) if sort_key is not None else meta.sequence
    return key, meta.sequence


def _completed_result(
    future: Future[_Result],
    meta: _ActiveTask[_Item],
    *,
    stage: str,
) -> _Result:
    try:
        return future.result(timeout=0)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise ParallelTaskError(stage=stage, item=meta.item, cause=exc) from exc


def _shutdown_pool(
    pool: ThreadPoolExecutor,
    active: dict[Future[_Result], _ActiveTask[_Item]],
) -> None:
    for future in active:
        future.cancel()
    active.clear()
    pool.shutdown(wait=False, cancel_futures=True)


@dataclass
class _DrainContext(Generic[_Item, _Result]):
    state: _SubmissionState[_Item]
    active: dict[Future[_Result], _ActiveTask[_Item]]
    pool: ThreadPoolExecutor
    worker: Callable[[_Item], _Result]
    workers: int
    unit_timeout: float
    stage_deadline: float | None
    stage: str
    sort_key: Callable[[_Item], object] | None
    on_result: Callable[[_Item, _Result], None] | None
    on_error: Callable[[_Item, BaseException], None] | None


def _drain_active_tasks(
    context: _DrainContext[_Item, _Result],
) -> Iterator[tuple[_Item, _Result]]:
    while context.active:
        try:
            done = _next_completed_batch(
                context.active,
                stage=context.stage,
                unit_timeout=context.unit_timeout,
                stage_deadline=context.stage_deadline,
            )
        except ParallelExecutionTimeout as exc:
            if context.on_error is None:
                raise
            context.on_error(exc.item, exc)
            return
        ordered = sorted(
            done,
            key=lambda future: _completion_order(
                future, context.active, context.sort_key
            ),
        )
        for future in ordered:
            meta = context.active.pop(future)
            try:
                result = _completed_result(future, meta, stage=context.stage)
            except ParallelTaskError as exc:
                if context.on_error is None:
                    raise
                context.on_error(meta.item, exc.cause)
                continue
            if context.on_result is not None:
                context.on_result(meta.item, result)
            yield meta.item, result
        _fill_active_slots(
            context.state,
            context.active,
            context.pool,
            context.worker,
            workers=context.workers,
            unit_timeout=context.unit_timeout,
            stage_deadline=context.stage_deadline,
        )


def _iter_completed_with_deadlines_impl(
    items: Iterable[_Item],
    worker: Callable[[_Item], _Result],
    *,
    max_workers: int,
    stage: str,
    sort_key: Callable[[_Item], object] | None,
    on_result: Callable[[_Item, _Result], None] | None,
    on_error: Callable[[_Item, BaseException], None] | None,
    work_unit_timeout_seconds: float | None,
) -> Iterator[tuple[_Item, _Result]]:
    total_units = len(items) if isinstance(items, Sized) else None
    if total_units is not None and total_units <= 0:
        return

    workers = _resolve_worker_count(total_units, max_workers)
    started_at = time.monotonic()
    unit_timeout = (
        planning_work_unit_timeout_seconds()
        if work_unit_timeout_seconds is None
        else max(0.001, float(work_unit_timeout_seconds))
    )
    stage_deadline = None
    if total_units is not None:
        if work_unit_timeout_seconds is None:
            stage_deadline = planning_stage_deadline(
                work_units=total_units,
                workers=workers,
                started_at=started_at,
            )
        else:
            waves = max(1, (total_units + workers - 1) // workers)
            stage_deadline = started_at + unit_timeout * waves
    state = _SubmissionState(iter(items))
    active: dict[Future[_Result], _ActiveTask[_Item]] = {}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=stage)

    try:
        _fill_active_slots(
            state,
            active,
            pool,
            worker,
            workers=workers,
            unit_timeout=unit_timeout,
            stage_deadline=stage_deadline,
        )
        context = _DrainContext(
            state=state,
            active=active,
            pool=pool,
            worker=worker,
            workers=workers,
            unit_timeout=unit_timeout,
            stage_deadline=stage_deadline,
            stage=stage,
            sort_key=sort_key,
            on_result=on_result,
            on_error=on_error,
            work_unit_timeout_seconds=work_unit_timeout_seconds,
        )
        yield from _drain_active_tasks(context)
    finally:
        _shutdown_pool(pool, active)


def iter_completed_with_deadlines(
    items: Iterable[_Item],
    worker: Callable[[_Item], _Result],
    *,
    max_workers: int,
    stage: str,
    sort_key: Callable[[_Item], object] | None = None,
    on_result: Callable[[_Item, _Result], None] | None = None,
    on_error: Callable[[_Item, BaseException], None] | None = None,
    work_unit_timeout_seconds: float | None = None,
) -> Iterator[tuple[_Item, _Result]]:
    """Yield completed work through a bounded deadline-aware submission window.

    ``on_error`` is an opt-in isolation boundary. Without it, worker errors and deadline
    expiry preserve the historical fail-fast behavior. With it, an individual worker
    exception is reported to the caller and independent work continues; deadline expiry
    is reported and ends only the current scheduling round so a durable caller can requeue
    unfinished work without converting the interruption into a terminal domain failure.
    """

    yield from _iter_completed_with_deadlines_impl(
        items,
        worker,
        max_workers=max_workers,
        stage=stage,
        sort_key=sort_key,
        on_result=on_result,
        on_error=on_error,
        work_unit_timeout_seconds=work_unit_timeout_seconds,
    )


def collect_completed_with_deadlines(
    items: Iterable[_Item],
    worker: Callable[[_Item], _Result],
    *,
    max_workers: int,
    stage: str,
    sort_key: Callable[[_Item], object] | None = None,
    on_result: Callable[[_Item, _Result], None] | None = None,
    on_error: Callable[[_Item, BaseException], None] | None = None,
    work_unit_timeout_seconds: float | None = None,
) -> list[tuple[_Item, _Result]]:
    """Compatibility collector for callers that explicitly need all results in memory."""

    return list(
        iter_completed_with_deadlines(
            items,
            worker,
            max_workers=max_workers,
            stage=stage,
            sort_key=sort_key,
            on_result=on_result,
            on_error=on_error,
        )
    )


__all__ = [
    "ParallelExecutionTimeout",
    "ParallelTaskError",
    "collect_completed_with_deadlines",
    "iter_completed_with_deadlines",
]
