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
    on_result: Callable[[_Item, _Result], None] | None = None,
) -> Iterator[tuple[_Item, _Result]]:
    """Yield completed work with a bounded submission/result-retention window.

    At most ``max_workers`` futures are alive at once. Completed values are yielded
    immediately after ``on_result`` has checkpointed them, and the executor keeps no
    historical completed-results list. This is the memory-bounded path for planning
    and research stages that may produce large model responses.

    Sized inputs retain the existing stage-deadline calculation without copying the
    entire input. Generic unsized iterables remain lazy and use the per-work-unit
    execution deadline because their total wave count is unknowable without defeating
    streaming by materializing the iterable.

    If a caller intentionally stops early, it should close the iterator so pending
    futures are cancelled immediately. Running Python threads cannot be forcibly
    killed, so transport/model adapters must still honor the propagated deadline.
    """

    total_units = len(items) if isinstance(items, Sized) else None
    if total_units is not None and total_units <= 0:
        return

    workers = max(1, int(max_workers))
    if total_units is not None:
        workers = min(workers, total_units)

    started_at = time.monotonic()
    unit_timeout = planning_work_unit_timeout_seconds()
    stage_deadline = (
        planning_stage_deadline(
            work_units=total_units,
            workers=workers,
            started_at=started_at,
        )
        if total_units is not None
        else None
    )
    source = iter(items)
    source_exhausted = False
    next_sequence = 0
    active: dict[Future[_Result], _ActiveTask[_Item]] = {}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=stage)

    def submit_next() -> bool:
        nonlocal source_exhausted, next_sequence
        if source_exhausted:
            return False
        try:
            item = next(source)
        except StopIteration:
            source_exhausted = True
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
            sequence=next_sequence,
            item=item,
            deadline=deadline,
            submitted_at=submitted_at,
        )
        next_sequence += 1
        return True

    try:
        while len(active) < workers and submit_next():
            pass

        while active:
            now = time.monotonic()
            nearest_task_deadline = min(meta.deadline for meta in active.values())
            wake_deadline = nearest_task_deadline
            if stage_deadline is not None:
                wake_deadline = min(stage_deadline, wake_deadline)
            timeout = max(0.0, wake_deadline - now)
            done, not_done = wait(
                tuple(active),
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                now = time.monotonic()
                deadline_kind = (
                    "stage"
                    if stage_deadline is not None and now >= stage_deadline
                    else "work_unit"
                )
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

            ordered = sorted(done, key=completion_order)
            for future in ordered:
                meta = active.pop(future)
                try:
                    result = future.result(timeout=0)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    raise ParallelTaskError(stage=stage, item=meta.item, cause=exc) from exc
                if on_result is not None:
                    on_result(meta.item, result)
                yield meta.item, result
                del result
                del meta

            del ordered
            del done
            del not_done

            while len(active) < workers and submit_next():
                pass
    finally:
        for future in active:
            future.cancel()
        active.clear()
        pool.shutdown(wait=False, cancel_futures=True)


def collect_completed_with_deadlines(
    items: Iterable[_Item],
    worker: Callable[[_Item], _Result],
    *,
    max_workers: int,
    stage: str,
    sort_key: Callable[[_Item], object] | None = None,
    on_result: Callable[[_Item, _Result], None] | None = None,
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
        )
    )


__all__ = [
    "ParallelExecutionTimeout",
    "ParallelTaskError",
    "collect_completed_with_deadlines",
    "iter_completed_with_deadlines",
]
