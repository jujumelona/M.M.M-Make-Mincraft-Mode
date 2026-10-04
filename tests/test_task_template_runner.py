from copy import deepcopy

import pytest

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai import task_template_runner as runner
from minecraft_mod_ai.task_template_catalog import ROOT, load_template


def drive(monkeypatch, records):
    count_calls = []
    record_contexts = []
    records = [deepcopy(record) for record in records]

    def choose_count(*args, **kwargs):
        del args
        count_calls.append(kwargs)
        return {"count": len(records)}

    def generate_record(
        model_router,
        identifier,
        *,
        context,
        progress,
        checkpoint,
    ):
        del model_router, identifier, progress, checkpoint
        record_contexts.append(deepcopy(context))
        return deepcopy(records[int(context["record_index"])])

    monkeypatch.setattr(bounded, "generate_fixed_template_value", choose_count)
    monkeypatch.setattr(bounded, "run_single_record_template", generate_record)
    return count_calls, record_contexts


def test_record_roundtrip_uses_count_then_host_owned_record_jobs(monkeypatch):
    record = {"trigger": "right click", "owner": "server player"}
    count_calls, record_contexts = drive(monkeypatch, [record])
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={"criterion": "activate"},
        allowed_refs=set(),
    )
    assert result["records"] == [record]
    assert len(count_calls) == 1
    schema = count_calls[0]["response_schema"]
    assert schema["required"] == ["count"]
    assert set(schema["properties"]) == {"count"}
    assert schema["properties"]["count"]["enum"] == list(range(17))
    assert len(record_contexts) == 1
    assert record_contexts[0]["record_index"] == 0
    assert record_contexts[0]["record_ordinal"] == 1
    assert record_contexts[0]["record_count"] == 1


def test_empty_result_is_host_owned_data_not_completion_protocol(monkeypatch):
    count_calls, record_contexts = drive(monkeypatch, [])
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs=set(),
    )
    assert result["records"] == []
    assert result["reason"]
    assert len(count_calls) == 1
    assert record_contexts == []


def test_duplicate_ordinal_records_are_rejected(monkeypatch):
    record = {"trigger": "right click", "owner": "server player"}
    drive(monkeypatch, [record, record])
    with pytest.raises(ValueError, match="TEMPLATE_RECORD_SET_DUPLICATE"):
        runner.run_record_template(
            None,
            "feature/behavior_contract/entry_conditions",
            context={},
            allowed_refs=set(),
        )


def test_record_set_schema_has_no_model_loop_control(monkeypatch):
    count_calls, _record_contexts = drive(monkeypatch, [])
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs=set(),
    )
    assert result["records"] == []
    schema = count_calls[0]["response_schema"]
    assert set(schema["properties"]) == {"count"}
    assert schema["required"] == ["count"]
    for forbidden in ("done", "next_work", "blocked_reason", "continuation", "records"):
        assert forbidden not in schema["properties"]


def test_catalog_manifests_resolve_every_declared_task():
    for path in ROOT.rglob("*.yaml"):
        task = load_template(path.relative_to(ROOT).with_suffix("").as_posix())
        for identifier in task.get("steps", []):
            assert load_template(identifier)["id"] == identifier


def test_allowed_evidence_is_not_automatically_attached(monkeypatch):
    record = {"trigger": "click", "owner": "server"}
    drive(monkeypatch, [record])
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs={"unrelated_a", "unrelated_b"},
    )
    assert result["evidence_refs"] == []


def test_host_context_can_admit_known_evidence(monkeypatch):
    record = {"trigger": "click", "owner": "server"}
    drive(monkeypatch, [record])
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={"evidence": ["e1", "not-allowed"]},
        allowed_refs={"e1"},
    )
    assert result["evidence_refs"] == ["e1"]


def test_entry_condition_record_contract_exposes_only_bounded_count_decision():
    schema = runner.record_response_schema(
        load_template("feature/behavior_contract/entry_conditions")
    )
    assert schema["required"] == ["count"]
    assert set(schema["properties"]) == {"count"}
    assert schema["properties"]["count"]["enum"] == list(range(17))
    assert schema["additionalProperties"] is False
