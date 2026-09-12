from __future__ import annotations

from minecraft_mod_ai import task_template_batch_runner as runner


def test_parallel_batch_uses_deadline_scheduler_and_restores_input_order(monkeypatch) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(runner, "router_native_model_parallelism", lambda model_router: 2)

    def fake_record_template(
        model_router,
        identifier,
        *,
        context,
        allowed_refs,
        progress,
        checkpoint,
    ):
        del model_router, context, allowed_refs, progress, checkpoint
        return {"identifier": identifier}

    monkeypatch.setattr(runner, "run_bounded_record_template", fake_record_template)

    def fake_deadline_scheduler(items, worker, *, max_workers, stage, sort_key=None):
        values = tuple(items)
        observed["items"] = values
        observed["max_workers"] = max_workers
        observed["stage"] = stage
        observed["sort_key"] = sort_key
        for identifier in reversed(values):
            yield identifier, worker(identifier)

    monkeypatch.setattr(runner, "iter_completed_with_deadlines", fake_deadline_scheduler)

    result = runner.run_record_template_batch(
        object(),
        ("req_001", "req_002"),
        context={"scope": "test"},
        allowed_refs={"ref_1"},
    )

    assert observed == {
        "items": ("req_001", "req_002"),
        "max_workers": 2,
        "stage": "planning-template-record",
        "sort_key": None,
    }
    assert list(result) == ["req_001", "req_002"]
    assert result == {
        "req_001": {"identifier": "req_001"},
        "req_002": {"identifier": "req_002"},
    }
