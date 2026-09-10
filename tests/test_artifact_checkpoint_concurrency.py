from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.artifact_job_checkpoint import execute_checkpointed_job
from minecraft_mod_ai.artifact_ports import PortRegistry


def _job(job_id: str, target_path: str) -> ArtifactJob:
    return ArtifactJob(
        job_id=job_id,
        template_id="test/template",
        owner_module="test-module",
        target_path=target_path,
    )


def _receipt(job: ArtifactJob) -> dict[str, object]:
    return {
        "status": "PASS",
        "rendered_output": job.target_path,
        "validations": [],
        "ports_published": {},
    }


def test_checkpoint_allows_different_targets_to_execute_concurrently(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.artifact_job_checkpoint.load_template",
        lambda _template_id: {"schema": "test"},
    )
    entered = Barrier(2)
    release = Barrier(3)

    def execute(job, *, base_dir, **_kwargs):
        entered.wait(timeout=2)
        release.wait(timeout=2)
        target = base_dir / job.target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(job.job_id, encoding="utf-8")
        return _receipt(job)

    registry = PortRegistry()
    jobs = (_job("a", "out/a.txt"), _job("b", "out/b.txt"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                execute_checkpointed_job,
                job,
                context={},
                router=None,
                registry=registry,
                base_dir=tmp_path,
                execute=execute,
            )
            for job in jobs
        ]
        release.wait(timeout=2)
        receipts = [future.result(timeout=2) for future in futures]

    assert [receipt["status"] for receipt in receipts] == ["PASS", "PASS"]
    state = json.loads(
        (tmp_path / ".mmm/artifact_jobs.json").read_text(encoding="utf-8")
    )["state"]
    assert set(state["jobs"]) == {"a", "b"}
    assert set(state["paths"]) == {"out/a.txt", "out/b.txt"}


def test_checkpoint_serializes_jobs_that_share_a_target(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.artifact_job_checkpoint.load_template",
        lambda _template_id: {"schema": "test"},
    )
    first_entered = Event()
    second_call_started = Event()
    second_entered = Event()
    release_first = Event()

    def execute(job, *, base_dir, **_kwargs):
        if job.job_id == "first":
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        target = base_dir / job.target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(job.job_id, encoding="utf-8")
        return _receipt(job)

    registry = PortRegistry()
    first = _job("first", "out/shared.txt")
    second = _job("second", "out/shared.txt")

    def run_second():
        second_call_started.set()
        return execute_checkpointed_job(
            second,
            context={},
            router=None,
            registry=registry,
            base_dir=tmp_path,
            execute=execute,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            execute_checkpointed_job,
            first,
            context={},
            router=None,
            registry=registry,
            base_dir=tmp_path,
            execute=execute,
        )
        assert first_entered.wait(timeout=2)
        second_future = pool.submit(run_second)
        assert second_call_started.wait(timeout=2)
        assert not second_entered.wait(timeout=0.05)
        release_first.set()
        assert first_future.result(timeout=2)["status"] == "PASS"
        assert second_future.result(timeout=2)["status"] == "PASS"

    assert second_entered.is_set()
    assert (tmp_path / "out/shared.txt").read_text(encoding="utf-8") == "second"
    state = json.loads(
        (tmp_path / ".mmm/artifact_jobs.json").read_text(encoding="utf-8")
    )["state"]
    assert set(state["jobs"]) == {"first", "second"}
    assert state["paths"]["out/shared.txt"] == state["jobs"]["second"]["after_hash"]


def test_checkpoint_reuse_does_not_reexecute_target(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.artifact_job_checkpoint.load_template",
        lambda _template_id: {"schema": "test"},
    )
    calls = 0

    def execute(job, *, base_dir, **_kwargs):
        nonlocal calls
        calls += 1
        target = base_dir / job.target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stable", encoding="utf-8")
        return _receipt(job)

    registry = PortRegistry()
    first = _job("stable", "out/stable.txt")
    original = execute_checkpointed_job(
        first,
        context={},
        router=None,
        registry=registry,
        base_dir=tmp_path,
        execute=execute,
    )
    second = _job("stable", "out/stable.txt")
    reused = execute_checkpointed_job(
        second,
        context={},
        router=None,
        registry=registry,
        base_dir=tmp_path,
        execute=execute,
    )

    assert original["status"] == "PASS"
    assert reused["reuse"] == "VERIFIED_CHECKPOINT"
    assert calls == 1
