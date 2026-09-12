from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_pipeline import (
    PlanningStage,
    PlanningStageError,
    _bounded_future_result,
)


class _TimeoutFuture:
    def __init__(self) -> None:
        self.timeout: float | None = None
        self.cancelled = False

    def result(self, *, timeout: float):
        self.timeout = timeout
        raise TimeoutError("simulated blocked inventory scan")

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class _UnboundedOnlyFuture:
    def result(self):
        raise AssertionError("unbounded result() must never be called")


def test_bounded_future_result_times_out_and_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS", "0.125")
    future = _TimeoutFuture()

    with pytest.raises(PlanningStageError) as caught:
        _bounded_future_result(future, operation="existing project inventory")

    assert caught.value.stage is PlanningStage.DESIGN
    assert future.timeout == pytest.approx(0.125)
    assert future.cancelled is True
    assert "planning work-unit deadline" in str(caught.value)


def test_bounded_future_result_rejects_unbounded_future_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS", "0.125")

    with pytest.raises(PlanningStageError) as caught:
        _bounded_future_result(
            _UnboundedOnlyFuture(),
            operation="existing project inventory",
        )

    assert caught.value.stage is PlanningStage.DESIGN
    assert "does not support bounded result" in str(caught.value)
