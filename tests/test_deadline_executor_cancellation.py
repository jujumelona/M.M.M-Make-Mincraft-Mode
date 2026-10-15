"""The planner's executor must share the artifact executor's lifecycle contract."""
from __future__ import annotations

import threading
import time

import pytest

from minecraft_mod_ai.deadline_executor import iter_completed_with_deadlines
from minecraft_mod_ai.model_concurrency import (
    refresh_model_execution_deadline,
    remaining_model_execution_seconds,
)


def test_closing_result_iterator_cancels_running_planner_worker():
    started, release, stopped = threading.Event(), threading.Event(), threading.Event()
    proceeded, errors = [], []

    def work(item):
        if item == 'fast':
            assert started.wait(2)
            return item
        started.set()
        try:
            assert release.wait(2)
            remaining_model_execution_seconds()
            proceeded.append(item)
            return item
        except Exception as exc:
            errors.append(exc)
            raise
        finally:
            stopped.set()

    results = iter_completed_with_deadlines(['slow', 'fast'], work,
        max_workers=2, stage='planner-cancellation', work_unit_timeout_seconds=5)
    try:
        assert next(results) == ('fast', 'fast')
        results.close()
    finally:
        release.set()
        assert stopped.wait(2)
    assert not proceeded
    assert errors and 'cancel' in str(errors[0]).lower()


@pytest.mark.parametrize('items', [['slow'], ('slow',), (item for item in ['slow'])])
def test_streamed_work_observes_progress_refreshed_lease(items):
    def work(item):
        until = time.monotonic() + 0.18
        while time.monotonic() < until:
            refresh_model_execution_deadline(0.12)
            time.sleep(0.01)
        return item

    results = list(iter_completed_with_deadlines(items, work,
        max_workers=1, stage='planner-progress', work_unit_timeout_seconds=0.08))
    assert results == [('slow', 'slow')]


def test_new_work_wave_does_not_inherit_expired_initial_stage_deadline():
    def work(item):
        if item == 'slow':
            until = time.monotonic() + 0.20
            while time.monotonic() < until:
                refresh_model_execution_deadline(0.12)
                time.sleep(0.01)
        assert remaining_model_execution_seconds() > 0
        return item

    results = list(iter_completed_with_deadlines(['slow', 'next'], work,
        max_workers=1, stage='planner-waves', work_unit_timeout_seconds=0.08))
    assert results == [('slow', 'slow'), ('next', 'next')]
