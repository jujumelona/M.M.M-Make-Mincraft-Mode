from __future__ import annotations

"""Deadline-aware parallel execution for model-backed planning work.

The executor is intentionally small: callers retain semantic ownership of retries,
checkpoints, and failure policy. This module owns only bounded scheduling, monotonic
work/stage deadlines, context propagation, cancellation, and non-blocking shutdown.
"""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
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


def iter_completed_with_deadlines(
    items: Iterable[_Item],
    worker: Callable[[_Item], _Result],
    *,
    max_workers: int,
    stage: str,
    sort_key: Callable[[_Item], object] | None = None,
) -> Iterator[tuple[_Item, _Result]]:
    """Yield completed work while enforcing per-unit and whole-stage deadlines.

    Only active work is submitted, so queued work never consumes its timeout before it
    actually owns an executor slot. Every worker receives the same absolute deadline in
    its copied context, allowing model capacity locks to use the remaining budget too.
    Executor cleanup never waits for an uncooperative thread; transport adapters remain
    responsible for their own finite I/O timeout.
    """

    indexed = list(enumerate(items))
    if not indexed:
        return

    workers = max(1, min(int(max_workers), len(indexed)))
    started_at = time.monotonic()
    unit_timeout = planning_work_unit_timeout_seconds()
    stage_deadline = planning_stage_deadline(
        work_units=len(indexed),
        workers=workers,
        started_at=started_at,
    )
    pending = deque(indexed)
    active: dict[Future[_Result], _ActiveTask[_Item]] = {}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=stage)

    def submit_one(sequence: int, item: _Item) -> None:
        submitted_at = time.monotonic()
        deadline = min(stage_deadline, submitted_at + unit_timeout)
        context = copy_context()
        future = pool.submit(
            context.run,
            run_with_model_execution_deadline,
            deadline,
            worker,
            item,
        )
        active[future] = _ActiveTask(
            sequence=sequence,
            item=item,
            deadline=deadline,
            submitted_at=submitted_at,
        )

    try:
        while pending or active:
            while pending and len(active) < workers:
                sequence, item = pending.popleft()
                submit_one(sequence, item)

            if not active:
                break

            now = time.monotonic()
            nearest_task_deadline = min(meta.deadline for meta in active.values())
            wake_deadline = min(stage_deadline, nearest_task_deadline)
            timeout = max(0.0, wake_deadline - now)
            done, _ = wait(
                tuple(active),
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                now = time.monotonic()
                deadline_kind = "stage" if now >= stage_deadline else "work_unit"
                expired = min(
                    active.values(),
                    key=lambda meta: (meta.deadline, meta.sequence),
                )
                raise ParallelExecutionTimeout(
                    stage=stage,
                    item=expired.item,
                    elapsed_seconds=now - expired.submitted_at,
                    work_unit_timeout_seconds=unit_timeout,
                    deadline_kind=deadline_kind,
                )

            def completion_order(future: Future[_Result]) -> tuple[object, int]:
                meta = active[future]
                key = sort_key(meta.item) if sort_key is not None else meta.sequence
                return key, meta.sequence

            for future in sorted(done, key=completion_order):
                meta = active.pop(future)
                try:
                    result = future.result(timeout=0)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    raise ParallelTaskError(stage=stage, item=meta.item, cause=exc) from exc
                yield meta.item, result
    finally:
        for future in active:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "ParallelExecutionTimeout",
    "ParallelTaskError",
    "iter_completed_with_deadlines",
]
