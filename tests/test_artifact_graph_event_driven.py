from __future__ import annotations

import threading

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.artifact_ports import PortKind, PortRegistry, TypedPort
import minecraft_mod_ai.artifact_graph_executor as executor


def _job(job_id: str, *, requires=(), produces=()) -> ArtifactJob:
    return ArtifactJob(
        job_id=job_id,
        template_id="test",
        owner_module="test",
        requires=tuple(requires),
        produces=tuple(produces),
    )


def _publish(registry: PortRegistry, name: str, job_id: str) -> None:
    registry.publish(
        TypedPort(
            name=name,
            port_kind=PortKind.GENERIC,
            target_type="Any",
            value=job_id,
        )
    )


def test_dependent_starts_before_unrelated_slow_sibling_finishes(monkeypatch):
    fast_done = threading.Event()
    slow_release = threading.Event()
    dependent_started = threading.Event()

    def fake_execute(job, *, registry, **kwargs):
        if job.job_id == "fast":
            _publish(registry, "fast.port", job.job_id)
            fast_done.set()
        elif job.job_id == "slow":
            assert fast_done.wait(2)
            assert dependent_started.wait(2)
            slow_release.set()
        elif job.job_id == "dependent":
            dependent_started.set()
        return {"status": "PASS"}

    monkeypatch.setattr(executor, "_execute_one", fake_execute)
    monkeypatch.setenv("MMM_ARTIFACT_MAX_WORKERS", "2")

    result = executor.execute_artifact_graph(
        [
            _job("fast", produces=("fast.port",)),
            _job("slow"),
            _job("dependent", requires=("fast.port",)),
        ]
    )

    assert slow_release.is_set()
    assert dependent_started.is_set()
    assert result["completed_jobs"] == ["fast", "slow", "dependent"]


def test_receipt_order_is_input_order_not_completion_order(monkeypatch):
    first_release = threading.Event()
    second_done = threading.Event()

    def fake_execute(job, **kwargs):
        if job.job_id == "first":
            assert second_done.wait(2)
            first_release.set()
        else:
            second_done.set()
        return {"status": "PASS", "job": job.job_id}

    monkeypatch.setattr(executor, "_execute_one", fake_execute)
    monkeypatch.setenv("MMM_ARTIFACT_MAX_WORKERS", "2")

    result = executor.execute_artifact_graph([_job("first"), _job("second")])
    assert first_release.is_set()
    assert [receipt["job"] for receipt in result["receipts"]] == ["first", "second"]


def test_worker_cap_limits_parallel_execution(monkeypatch):
    lock = threading.Lock()
    active = 0
    peak = 0
    gate = threading.Barrier(2)

    def fake_execute(job, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        gate.wait(timeout=2)
        with lock:
            active -= 1
        return {"status": "PASS"}

    monkeypatch.setattr(executor, "_execute_one", fake_execute)
    monkeypatch.setenv("MMM_ARTIFACT_MAX_WORKERS", "2")

    executor.execute_artifact_graph([_job("a"), _job("b")])
    assert peak == 2


def test_validation_failure_blocks_dependent(monkeypatch):
    dependent_started = threading.Event()

    def fake_execute(job, *, registry, **kwargs):
        if job.job_id == "producer":
            _publish(registry, "p", job.job_id)
        else:
            dependent_started.set()
        return {"status": "PASS"}

    def validator(job, receipt):
        return False if job.job_id == "producer" else True

    monkeypatch.setattr(executor, "_execute_one", fake_execute)
    monkeypatch.setenv("MMM_ARTIFACT_MAX_WORKERS", "1")

    try:
        executor.execute_artifact_graph(
            [_job("producer", produces=("p",)), _job("dependent", requires=("p",))],
            context={"artifact_validator": validator},
        )
    except executor.ArtifactGraphError as exc:
        assert "ARTIFACT_VALIDATION_FAILED" in str(exc)
    else:
        raise AssertionError("expected validation failure")
    assert not dependent_started.is_set()
