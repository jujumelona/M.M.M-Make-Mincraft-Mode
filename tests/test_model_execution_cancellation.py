"""Cancellation follows nested scopes, streaming responses, and queued requests."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.llama_completion_liveness_contract import _ProgressCheckedResponse
from minecraft_mod_ai.model_concurrency import (
    ModelExecutionBudget,
    ModelExecutionCancelled,
    ReentrantCapacityGate,
    bind_model_execution_deadline,
    current_model_execution_deadline,
    refresh_model_execution_deadline,
    remaining_model_execution_seconds,
)
from minecraft_mod_ai.model_router import ModelRouter


def test_nested_deadline_cannot_escape_parent_cancellation():
    budget = ModelExecutionBudget(time.monotonic() + 10)
    with bind_model_execution_deadline(budget), bind_model_execution_deadline(time.monotonic() + 5):
        budget.cancel()
        with pytest.raises(ModelExecutionCancelled):
            remaining_model_execution_seconds()
    assert current_model_execution_deadline() is None


def test_progress_renews_every_owning_work_budget():
    parent = ModelExecutionBudget(time.monotonic() + 1)
    with bind_model_execution_deadline(parent):
        child = ModelExecutionBudget(time.monotonic() + 0.5)
        with bind_model_execution_deadline(child):
            refreshed = refresh_model_execution_deadline(30)
            assert parent.deadline >= refreshed
            assert child.deadline >= refreshed


@pytest.mark.parametrize('cancel_after_response', [False, True])
def test_router_fences_cancelled_model_requests_and_late_responses(cancel_after_response):
    router = object.__new__(ModelRouter)
    config = SimpleNamespace(exclusive_gpu=False)
    budget = ModelExecutionBudget(time.monotonic() + 10)
    entered = []
    with bind_model_execution_deadline(budget):
        if not cancel_after_response:
            budget.cancel()
        with pytest.raises(ModelExecutionCancelled), router._generation_scope(config):
            entered.append('model request')
            budget.cancel()
    assert bool(entered) == cancel_after_response


def test_sse_stream_stops_before_accepting_another_cancelled_delta():
    budget = ModelExecutionBudget(time.monotonic() + 10)

    def lines():
        yield 'data: {"choices":[{"delta":{"content":"first"}}]}'
        budget.cancel()
        yield 'data: {"choices":[{"delta":{"content":"late"}}]}'

    wrapped = _ProgressCheckedResponse(SimpleNamespace(iter_lines=lines), 120,
        request_id='cancelled-stream', started_at=time.monotonic())
    with bind_model_execution_deadline(budget):
        iterator = wrapped.iter_lines()
        assert 'first' in next(iterator)
        with pytest.raises(ModelExecutionCancelled):
            next(iterator)


def test_capacity_wait_observes_cancellation_before_slot_owner_releases():
    gate = ReentrantCapacityGate(lambda: 1)
    budget = ModelExecutionBudget(time.monotonic() + 30)
    started, stopped = threading.Event(), threading.Event()
    errors, entered = [], []

    def wait_for_slot():
        try:
            with bind_model_execution_deadline(budget):
                started.set()
                with gate:
                    entered.append('slot')
        except ModelExecutionCancelled as exc:
            errors.append(exc)
        finally:
            stopped.set()

    with gate:
        thread = threading.Thread(target=wait_for_slot)
        thread.start()
        assert started.wait(2)
        budget.cancel()
        assert stopped.wait(1)
        assert not entered
    thread.join(2)
    assert len(errors) == 1 and isinstance(errors[0], ModelExecutionCancelled)
