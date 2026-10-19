from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai.deadline_executor import ParallelExecutionTimeout
from minecraft_mod_ai.execution_contract_policy import (
    PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING,
)
from minecraft_mod_ai.model_concurrency import (
    ModelExecutionBudget,
    ModelExecutionCancelled,
    bind_model_execution_deadline,
    current_model_execution_deadline,
    refresh_model_execution_deadline,
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
    assert [item["accepted_records"] for item in record_contexts] == [
        [],
        [{"name": "item-1"}],
    ]


def test_duplicate_is_rejected_before_publishing_or_authoring_later_ordinals(
    monkeypatch,
) -> None:
    saved = []
    contexts = []
    monkeypatch.setattr(bounded, "load_record_template", lambda _: deepcopy(_TEMPLATE))
    monkeypatch.setattr(
        bounded, "generate_fixed_template_value", lambda *_args, **_kwargs: {"count": 3}
    )

    def duplicate_record(_router, _identifier, *, context, **_kwargs):
        contexts.append(deepcopy(context))
        return {"name": "same"}

    monkeypatch.setattr(bounded, "run_single_record_template", duplicate_record)
    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_DUPLICATE"):
        bounded.run_bounded_record_template(
            object(),
            "feature/test/items",
            context={},
            checkpoint=lambda key, value: saved.append((key, deepcopy(value))),
        )

    assert [context["record_index"] for context in contexts] == [0, 1]
    assert all("records" not in value for _key, value in saved)


def test_accepted_prefix_is_an_owned_snapshot(monkeypatch) -> None:
    contexts = []
    monkeypatch.setattr(bounded, "load_record_template", lambda _: deepcopy(_TEMPLATE))
    monkeypatch.setattr(
        bounded, "generate_fixed_template_value", lambda *_args, **_kwargs: {"count": 2}
    )

    def author_record(_router, _identifier, *, context, **_kwargs):
        contexts.append(deepcopy(context))
        if context["accepted_records"]:
            context["accepted_records"][0]["name"] = "mutated"
        return {"name": f"item-{context['record_ordinal']}"}

    monkeypatch.setattr(bounded, "run_single_record_template", author_record)
    result = bounded.run_bounded_record_template(
        object(), "feature/test/items", context={}
    )
    assert result["records"] == [{"name": "item-1"}, {"name": "item-2"}]
    assert contexts[1]["accepted_records"] == [{"name": "item-1"}]


def test_each_dependent_ordinal_keeps_its_own_renewable_execution_lease(monkeypatch):
    monkeypatch.setattr(bounded, "load_record_template", lambda _: deepcopy(_TEMPLATE))
    monkeypatch.setattr(
        bounded, "generate_fixed_template_value", lambda *_args, **_kwargs: {"count": 2}
    )
    leases = []

    def author_record(_router, _identifier, *, context, **_kwargs):
        deadline = current_model_execution_deadline()
        assert deadline is not None
        renewed = refresh_model_execution_deadline(1200)
        leases.append((deadline, renewed))
        return {"name": f"item-{context['record_ordinal']}"}

    monkeypatch.setattr(bounded, "run_single_record_template", author_record)
    bounded.run_bounded_record_template(object(), "feature/test/items", context={})
    assert len(leases) == 2
    assert all(renewed > original for original, renewed in leases)
    assert leases[1][0] < leases[0][1]
    assert current_model_execution_deadline() is None


def test_abandoned_ordinal_cannot_publish_or_start_later_records(monkeypatch):
    release = threading.Event()
    worker_finished = threading.Event()
    ordinals = []
    saved = []
    monkeypatch.setenv("MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setattr(bounded, "load_record_template", lambda _: deepcopy(_TEMPLATE))
    monkeypatch.setattr(
        bounded, "generate_fixed_template_value", lambda *_args, **_kwargs: {"count": 2}
    )

    def blocked_record(_router, _identifier, *, context, checkpoint, **_kwargs):
        assert current_model_execution_deadline() is not None
        ordinals.append(context["record_index"])
        assert release.wait(2)
        candidate = {"name": "late"}
        checkpoint("single:late", candidate)
        worker_finished.set()
        return candidate

    monkeypatch.setattr(bounded, "run_single_record_template", blocked_record)
    try:
        with pytest.raises(ParallelExecutionTimeout):
            bounded.run_bounded_record_template(
                object(),
                "feature/test/items",
                context={},
                checkpoint=lambda key, value: saved.append((key, deepcopy(value))),
            )
    finally:
        release.set()
        assert worker_finished.wait(2)
    assert ordinals == [0]
    assert len(saved) == 1
    assert saved[0][1] == {"count": 2}


def test_admitted_checkpoints_run_on_caller_outside_model_worker_lease(monkeypatch):
    caller = threading.get_ident()
    saved = []
    monkeypatch.setattr(bounded, "load_record_template", lambda _: deepcopy(_TEMPLATE))
    monkeypatch.setattr(
        bounded, "generate_fixed_template_value", lambda *_args, **_kwargs: {"count": 2}
    )

    def author_record(_router, _identifier, *, context, checkpoint, **_kwargs):
        assert current_model_execution_deadline() is not None
        record = {"name": f"item-{context['record_ordinal']}"}
        checkpoint(f"single:{context['record_index']}", record)
        return record

    def checkpoint(key, value):
        assert threading.get_ident() == caller
        assert current_model_execution_deadline() is None
        saved.append((key, deepcopy(value)))

    monkeypatch.setattr(bounded, "run_single_record_template", author_record)
    result = bounded.run_bounded_record_template(
        object(), "feature/test/items", context={}, checkpoint=checkpoint
    )
    assert [key for key, _value in saved if key.startswith("single:")] == [
        "single:0", "single:1"
    ]
    assert saved[-1][1]["records"] == result["records"]


@pytest.mark.parametrize("cancel_at", ["single", "complete"])
def test_parent_cancellation_during_last_checkpoint_cannot_return_success(
    monkeypatch, cancel_at
):
    parent = ModelExecutionBudget(time.monotonic() + 30)
    saved = []
    monkeypatch.setattr(bounded, "load_record_template", lambda _: deepcopy(_TEMPLATE))
    monkeypatch.setattr(
        bounded, "generate_fixed_template_value", lambda *_args, **_kwargs: {"count": 1}
    )

    def author_record(_router, _identifier, *, checkpoint, **_kwargs):
        candidate = {"name": "item-one"}
        checkpoint("single:one", candidate)
        return candidate

    def checkpoint(key, value):
        saved.append((key, deepcopy(value)))
        if (cancel_at == "single" and key.startswith("single:")) or (
            cancel_at == "complete" and "records" in value
        ):
            parent.cancel()

    monkeypatch.setattr(bounded, "run_single_record_template", author_record)
    with bind_model_execution_deadline(parent), pytest.raises(ModelExecutionCancelled):
        bounded.run_bounded_record_template(
            object(), "feature/test/items", context={}, checkpoint=checkpoint
        )
    if cancel_at == "single":
        assert all("records" not in value for _key, value in saved)


def test_cancelled_parent_cannot_return_cached_complete_record_set(monkeypatch):
    monkeypatch.setattr(bounded, "load_record_template", lambda _: deepcopy(_TEMPLATE))
    monkeypatch.setattr(bounded, "task_binding", lambda *_args: "binding")
    parent = ModelExecutionBudget(time.monotonic() + 30)
    with bind_model_execution_deadline(parent):
        parent.cancel()
        with pytest.raises(ModelExecutionCancelled):
            bounded.run_bounded_record_template(
                object(),
                "feature/test/items",
                context={},
                progress={
                    "record-set-v4:min=0:binding": {
                        "count": 1, "records": [{"name": "item-one"}]
                    }
                },
            )


def test_legacy_duplicate_aggregate_does_not_poison_new_authoring(monkeypatch) -> None:
    monkeypatch.setattr(bounded, "load_record_template", lambda _: deepcopy(_TEMPLATE))
    monkeypatch.setattr(bounded, "task_binding", lambda *_args: "binding")
    monkeypatch.setattr(
        bounded, "generate_fixed_template_value", lambda *_args, **_kwargs: {"count": 2}
    )
    contexts = []

    def author_record(_router, _identifier, *, context, **_kwargs):
        contexts.append(deepcopy(context))
        return {"name": f"item-{len(context['accepted_records']) + 1}"}

    monkeypatch.setattr(bounded, "run_single_record_template", author_record)
    progress = {
        "record-set-v3:min=0:binding": {
            "count": 2,
            "records": [{"name": "same"}, {"name": "same"}],
        }
    }
    result = bounded.run_bounded_record_template(
        object(),
        "feature/test/items",
        context={},
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )
    assert result["records"] == [{"name": "item-1"}, {"name": "item-2"}]
    assert len(contexts) == 2
    assert progress["record-set-v4:min=0:binding"]["records"] == result["records"]


def test_real_field_authoring_uses_prefix_to_resolve_distinct_entities() -> None:
    expected = [
        {"role": "Smelt copper ore.", "kind": "item", "entity_id": "copper_ore"},
        {"role": "Power the spacecraft.", "kind": "item", "entity_id": "fuel_cell"},
        {"role": "Unlock the hangar.", "kind": "item", "entity_id": "red_key"},
    ]

    class PrefixRouter(_PlannerRouter):
        def generate_text(self, role, messages, **kwargs):
            assert role == "planner"
            schema = kwargs["response_schema"]
            fields = list(schema["properties"])
            self.text_calls.append({"messages": deepcopy(messages), "schema": schema})
            if fields == ["count"]:
                return json.dumps({"count": len(expected)})
            context = json.loads(messages[1]["content"])
            accepted = context["accepted_records"]
            # Author the complete bounded page. The singleton "kind" is
            # deterministic and inserted by the host without model inference.
            return json.dumps({
                field: expected[len(accepted)][field] for field in fields
            })

    router = PrefixRouter()
    result = bounded.run_bounded_record_template(
        router,
        "design/content_entity",
        context={
            "requirement": "Add copper ore, a fuel cell, and a red hangar key.",
            "allowed_content_kinds": ["item"],
        },
    )
    assert result["records"] == expected
    assert len(router.text_calls) == 1 + len(expected)
    for index, call in enumerate(router.text_calls[1:]):
        assert call["schema"]["required"] == ["role", "entity_id"]
        context = json.loads(call["messages"][1]["content"])
        assert context["accepted_records"] == expected[:index]
        assert "still-uncovered" in call["messages"][0]["content"]


@pytest.mark.parametrize("resumed", [False, True])
def test_entity_id_cannot_repeat_with_changed_description(monkeypatch, resumed) -> None:
    records = [
        {"role": "Unlock the hangar.", "kind": "item", "entity_id": "red_key"},
        {"role": "A key for the hangar.", "kind": "item", "entity_id": "red_key"},
    ]
    monkeypatch.setattr(bounded, "task_binding", lambda *_args: "binding")
    monkeypatch.setattr(
        bounded, "generate_fixed_template_value", lambda *_args, **_kwargs: {"count": 2}
    )
    monkeypatch.setattr(
        bounded,
        "run_single_record_template",
        lambda _router, _identifier, *, context, **_kwargs: records[context["record_index"]],
    )
    progress = (
        {"record-set-v4:min=0:binding": {"count": 2, "records": records}}
        if resumed else {}
    )
    saved = []
    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_DUPLICATE"):
        bounded.run_bounded_record_template(
            object(),
            "design/content_entity",
            context={},
            progress=progress,
            checkpoint=lambda _key, value: saved.append(deepcopy(value)),
        )
    assert all("records" not in value for value in saved)


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
        "record-set-v4:min=0:binding": {
            "count": len(records),
            "records": records,
        }
    }

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_COUNT"):
        bounded.run_bounded_record_template(
            object(),
            "feature/test/items",
            context={"requirement": "resume an oversized record set"},
            progress=progress,
        )

def test_record_minimum_above_host_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_MINIMUM"):
        bounded.record_cardinality_response_schema(minimum_count=17)

def test_resumed_duplicate_records_are_rejected(monkeypatch) -> None:
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

    progress = {
        "record-set-v4:min=0:binding": {
            "count": 2,
            "records": [{"name": "same"}, {"name": "same"}],
        }
    }

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_DUPLICATE"):
        bounded.run_bounded_record_template(
            object(),
            "feature/test/items",
            context={"requirement": "resume duplicate records"},
            progress=progress,
        )

def test_resumed_record_checkpoint_requires_integer_count(monkeypatch) -> None:
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

    for saved_records in ([], [{"name": "item-1"}]):
        progress = {
            "record-set-v4:min=0:binding": {
                "records": saved_records,
            }
        }
        with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_SAVED_COUNT"):
            bounded.run_bounded_record_template(
                object(),
                "feature/test/items",
                context={"requirement": "resume malformed checkpoint"},
                progress=progress,
            )

def test_malformed_record_checkpoint_shape_is_rejected(monkeypatch) -> None:
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

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_SAVED_SHAPE"):
        bounded.run_bounded_record_template(
            object(),
            "feature/test/items",
            context={"requirement": "resume invalid checkpoint"},
            progress={"record-set-v4:min=0:binding": "not-an-object"},
        )

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_SAVED_SHAPE"):
        bounded.run_bounded_record_template(
            object(),
            "feature/test/items",
            context={"requirement": "resume invalid records"},
            progress={
                "record-set-v4:min=0:binding": {
                    "count": 1,
                    "records": "not-a-list",
                }
            },
        )

def test_partial_record_checkpoint_count_must_be_integer(monkeypatch) -> None:
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

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_SAVED_COUNT"):
        bounded.run_bounded_record_template(
            object(),
            "feature/test/items",
            context={"requirement": "resume malformed partial checkpoint"},
            progress={"record-set-v4:min=0:binding": {"count": True}},
        )

def test_record_checkpoint_rejects_undeclared_fields(monkeypatch) -> None:
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

    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_SAVED_SHAPE"):
        bounded.run_bounded_record_template(
            object(),
            "feature/test/items",
            context={"requirement": "resume checkpoint with undeclared metadata"},
            progress={
                "record-set-v4:min=0:binding": {
                    "count": 1,
                    "debug": "unexpected",
                }
            },
        )

