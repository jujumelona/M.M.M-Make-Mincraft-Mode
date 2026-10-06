from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai.execution_contract_policy import (
    PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING,
)
from minecraft_mod_ai.model_output_atomicity_contract import (
    structured_output_token_ceiling,
)


_TEMPLATE = {
    "id": "feature/test/items",
    "task": "Author the required semantic items.",
    "rules": ["Keep each item distinct."],
    "record_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 32},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}


class _Registry:
    @staticmethod
    def role(_profile: str, _role: str):
        # Not an exclusive local model, so deterministic_model_map remains serial
        # in this unit test while preserving the production scheduling contract.
        return SimpleNamespace(
            adapter="llama_cpp",
            provider="local",
            exclusive_gpu=False,
        )


class _PlannerRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self) -> None:
        self.text_calls: list[dict] = []
        self.tool_calls = 0

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        assert kwargs["enable_tools"] is False
        assert kwargs["force_non_thinking"] is True
        schema = kwargs["response_schema"]
        assert kwargs["output_token_ceiling"] == structured_output_token_ceiling(schema)
        assert (
            kwargs["output_token_ceiling"]
            <= PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING
        )
        assert set(schema["properties"]) == {"count"}
        assert schema["required"] == ["count"]
        assert "records" not in schema["properties"]
        self.text_calls.append(
            {
                "messages": tuple(messages),
                "schema": schema,
                "kwargs": dict(kwargs),
            }
        )
        return json.dumps({"count": 2})

    def generate_tool_decision(self, *_args, **_kwargs):
        self.tool_calls += 1
        raise AssertionError("bounded planner records must never use native tools")


def test_bounded_record_set_is_count_then_host_owned_ordinals(monkeypatch) -> None:
    router = _PlannerRouter()
    record_contexts: list[dict] = []

    monkeypatch.setattr(
        bounded,
        "load_record_template",
        lambda identifier: dict(_TEMPLATE),
    )
    monkeypatch.setattr(
        bounded,
        "task_context",
        lambda template, context: dict(context),
    )

    def fake_single_record(
        model_router,
        identifier,
        *,
        context,
        progress,
        checkpoint,
    ):
        assert model_router is router
        assert identifier == "feature/test/items"
        assert progress is None
        assert checkpoint is None
        record_contexts.append(dict(context))
        index = int(context["record_index"])
        return {"name": f"item-{index + 1}"}

    monkeypatch.setattr(
        bounded,
        "run_single_record_template",
        fake_single_record,
    )

    result = bounded.run_bounded_record_template(
        router,
        "feature/test/items",
        context={"requirement": "make two distinct items"},
    )

    assert router.tool_calls == 0
    assert len(router.text_calls) == 1
    assert result["records"] == [
        {"name": "item-1"},
        {"name": "item-2"},
    ]
    assert result["reason"] == ""
    assert [item["record_index"] for item in record_contexts] == [0, 1]
    assert [item["record_ordinal"] for item in record_contexts] == [1, 2]
    assert all(item["record_count"] == 2 for item in record_contexts)
    assert all(item["accepted_records"] == [] for item in record_contexts)


def test_zero_count_performs_no_record_generation(monkeypatch) -> None:
    router = _PlannerRouter()

    monkeypatch.setattr(
        bounded,
        "load_record_template",
        lambda identifier: dict(_TEMPLATE),
    )
    monkeypatch.setattr(
        bounded,
        "task_context",
        lambda template, context: dict(context),
    )

    def zero_count(role, messages, **kwargs):
        assert role == "planner"
        assert set(kwargs["response_schema"]["properties"]) == {"count"}
        router.text_calls.append({"messages": tuple(messages), "kwargs": dict(kwargs)})
        return json.dumps({"count": 0})

    router.generate_text = zero_count

    def must_not_generate(*_args, **_kwargs):
        raise AssertionError("count=0 must not schedule a record model call")

    monkeypatch.setattr(
        bounded,
        "run_single_record_template",
        must_not_generate,
    )

    result = bounded.run_bounded_record_template(
        router,
        "feature/test/items",
        context={"requirement": "nothing applies"},
    )

    assert result["records"] == []
    assert result["reason"]
    assert router.tool_calls == 0


def test_record_cardinality_schema_is_finite_enum() -> None:
    schema = bounded.record_cardinality_response_schema()
    count = schema["properties"]["count"]
    assert count["minimum"] == 0
    assert count["maximum"] == 16
    assert count["enum"] == list(range(17))


def test_duplicate_ordinal_records_are_rejected(monkeypatch) -> None:
    router = _PlannerRouter()
    monkeypatch.setattr(
        bounded,
        "load_record_template",
        lambda identifier: dict(_TEMPLATE),
    )
    monkeypatch.setattr(
        bounded,
        "task_context",
        lambda template, context: dict(context),
    )

    def duplicate_record(
        model_router,
        identifier,
        *,
        context,
        progress,
        checkpoint,
    ):
        del model_router, identifier, context, progress, checkpoint
        return {"name": "same"}

    monkeypatch.setattr(
        bounded,
        "run_single_record_template",
        duplicate_record,
    )

    import pytest

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_DUPLICATE"):
        bounded.run_bounded_record_template(
            router,
            "feature/test/items",
            context={"requirement": "make two distinct items"},
        )

def test_saved_record_set_cannot_bypass_fresh_cardinality_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        bounded,
        "load_record_template",
        lambda identifier: dict(_TEMPLATE),
    )
    monkeypatch.setattr(
        bounded,
        "task_context",
        lambda template, context: dict(context),
    )
    monkeypatch.setattr(
        bounded,
        "task_binding",
        lambda template, context, refs: "binding",
    )

    records = [{"name": f"item-{index}"} for index in range(17)]
    progress = {
        "record-set-v3:min=0:binding": {
            "count": len(records),
            "records": records,
        }
    }

    import pytest

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_COUNT"):
        bounded.run_bounded_record_template(
            object(),
            "feature/test/items",
            context={"requirement": "resume an oversized record set"},
            progress=progress,
        )

def test_record_minimum_above_host_bound_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_MINIMUM"):
        bounded.record_cardinality_response_schema(minimum_count=17)

