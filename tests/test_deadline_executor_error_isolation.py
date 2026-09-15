from __future__ import annotations

import pytest

from minecraft_mod_ai.deadline_executor import (
    ParallelTaskError,
    iter_completed_with_deadlines,
)


def test_worker_error_callback_isolates_failed_item_and_continues_other_work() -> None:
    errors: list[tuple[int, str, str]] = []

    def worker(item: int) -> int:
        if item == 1:
            raise ValueError("recoverable model worker defect")
        return item * 10

    completed = list(
        iter_completed_with_deadlines(
            [1, 2, 3],
            worker,
            max_workers=2,
            stage="planning-test",
            sort_key=lambda item: item,
            on_error=lambda item, exc: errors.append(
                (item, type(exc).__name__, str(exc))
            ),
        )
    )

    assert sorted(completed) == [(2, 20), (3, 30)]
    assert errors == [(1, "ValueError", "recoverable model worker defect")]


def test_scheduler_remains_fail_fast_without_error_callback() -> None:
    def worker(item: int) -> int:
        if item == 1:
            raise ValueError("caller owns failure policy")
        return item

    with pytest.raises(ParallelTaskError):
        list(
            iter_completed_with_deadlines(
                [1],
                worker,
                max_workers=1,
                stage="generic-test",
            )
        )
