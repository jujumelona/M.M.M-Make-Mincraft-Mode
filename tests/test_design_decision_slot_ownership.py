from __future__ import annotations

from minecraft_mod_ai import design_record_runtime as runtime


def test_design_decisions_use_host_owned_singleton_slots(monkeypatch) -> None:
    seen_contexts: list[dict] = []
    captured_selector_schemas: list[dict] = []

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
        captured_selector_schemas.append(response_schema)
        return {
            "core_loop": True,
            "reward": True,
            "risk": False,
        }

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

    assert len(captured_selector_schemas) == 1
    assert captured_selector_schemas[0]["properties"] == {
        "core_loop": {"type": "boolean"},
        "reward": {"type": "boolean"},
        "risk": {"type": "boolean"},
    }
    assert captured_selector_schemas[0]["required"] == [
        "core_loop",
        "reward",
        "risk",
    ]
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


def test_slot_selection_cannot_duplicate_host_owned_slot_identity(monkeypatch) -> None:
    seen_contexts: list[dict] = []

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
    monkeypatch.setattr(
        runtime,
        "generate_fixed_template_value",
        lambda *_args, **_kwargs: {
            "reward": True,
            "risk": True,
        },
    )

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
        return {"slot_id": slot_id, "value": slot_id}

    monkeypatch.setattr(runtime, "run_single_record_template", fake_single)

    result = runtime.run_record_template(
        object(),
        "design/decision",
        context={
            "requirement_id": "req-1",
            "requirement": "Add reward and risk.",
            "allowed_slots": ["reward", "risk"],
        },
    )

    assert [row["slot_id"] for row in result["records"]] == ["reward", "risk"]
    assert [ctx["allowed_slots"] for ctx in seen_contexts] == [
        ["reward"],
        ["risk"],
    ]



def test_more_than_sixteen_candidate_slots_are_valid(monkeypatch) -> None:
    allowed_slots = [f"slot_{index}" for index in range(24)]
    selector_calls: list[dict] = []

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
        selector_calls.append(response_schema)
        return {
            slot_id: slot_id in {"slot_0", "slot_23"}
            for slot_id in allowed_slots
        }

    monkeypatch.setattr(runtime, "generate_fixed_template_value", fake_generate)
    monkeypatch.setattr(
        runtime,
        "run_single_record_template",
        lambda _router, _identifier, *, context, **_kwargs: {
            "slot_id": context["allowed_slots"][0],
            "value": context["allowed_slots"][0],
        },
    )

    result = runtime.run_record_template(
        object(),
        "design/decision",
        context={
            "requirement_id": "req-many-slots",
            "requirement": "Resolve a large host-owned design vocabulary.",
            "allowed_slots": allowed_slots,
        },
    )

    assert len(selector_calls) == 1
    assert set(selector_calls[0]["properties"]) == set(allowed_slots)
    assert result["records"] == [
        {"slot_id": "slot_0", "value": "slot_0"},
        {"slot_id": "slot_23", "value": "slot_23"},
    ]
