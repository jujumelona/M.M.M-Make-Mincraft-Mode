from __future__ import annotations

from minecraft_mod_ai import design_record_runtime as runtime


def test_design_decisions_use_host_owned_singleton_slots(monkeypatch) -> None:
    seen_contexts: list[dict] = []
    captured_selector_schema: dict = {}

    monkeypatch.setattr(
        runtime,
        "load_record_template",
        lambda _identifier: {
            "id": "design/decision",
            "task": "Resolve one decision.",
            "rules": [],
            "record_schema": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["slot_id", "value"],
                "additionalProperties": False,
            },
        },
    )
    monkeypatch.setattr(runtime, "task_context", lambda _template, context: dict(context))
    monkeypatch.setattr(runtime, "task_binding", lambda *_args, **_kwargs: "binding")
    monkeypatch.setattr(
        runtime,
        "deterministic_model_map",
        lambda _router, jobs, fn, **_kwargs: [fn(job) for job in jobs],
    )

    def fake_generate(
        _router,
        _role,
        _messages,
        *,
        response_schema,
        **_kwargs,
    ):
        captured_selector_schema.update(response_schema)
        return {"slots": ["core_loop", "reward"]}

    monkeypatch.setattr(runtime, "generate_fixed_template_value", fake_generate)

    def fake_single(
        _router,
        _identifier,
        *,
        context,
        progress,
        checkpoint,
    ):
        del progress, checkpoint
        seen_contexts.append(dict(context))
        slot_id = context["allowed_slots"][0]
        return {"slot_id": slot_id, "value": f"value for {slot_id}"}

    monkeypatch.setattr(runtime, "run_single_record_template", fake_single)

    result = runtime.run_record_template(
        object(),
        "design/decision",
        context={
            "requirement_id": "req-1",
            "requirement": "Add a reward-driven core loop.",
            "allowed_slots": ["core_loop", "reward", "risk"],
        },
    )

    assert captured_selector_schema["properties"]["slots"]["uniqueItems"] is True
    assert result["records"] == [
        {"slot_id": "core_loop", "value": "value for core_loop"},
        {"slot_id": "reward", "value": "value for reward"},
    ]
    assert [ctx["allowed_slots"] for ctx in seen_contexts] == [
        ["core_loop"],
        ["reward"],
    ]
    assert [ctx["record_ordinal"] for ctx in seen_contexts] == [1, 2]
    assert all(ctx["record_count"] == 2 for ctx in seen_contexts)


def test_duplicate_slot_selection_is_rejected_before_record_authoring(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "load_record_template",
        lambda _identifier: {
            "id": "design/decision",
            "task": "Resolve one decision.",
            "rules": [],
            "record_schema": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["slot_id", "value"],
                "additionalProperties": False,
            },
        },
    )
    monkeypatch.setattr(runtime, "task_context", lambda _template, context: dict(context))
    monkeypatch.setattr(runtime, "task_binding", lambda *_args, **_kwargs: "binding")
    monkeypatch.setattr(
        runtime,
        "generate_fixed_template_value",
        lambda *_args, **_kwargs: {"slots": ["reward", "reward"]},
    )

    def must_not_author(*_args, **_kwargs):
        raise AssertionError("duplicate slot selection must fail before record authoring")

    monkeypatch.setattr(runtime, "run_single_record_template", must_not_author)

    import pytest

    with pytest.raises(runtime.TemplateBlocked, match="TEMPLATE_DECISION_SLOT_SELECTION_INVALID"):
        runtime.run_record_template(
            object(),
            "design/decision",
            context={
                "requirement_id": "req-1",
                "requirement": "Add a reward.",
                "allowed_slots": ["reward", "risk"],
            },
        )
