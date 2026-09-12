from copy import deepcopy

import pytest

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai import task_template_runner as runner
from minecraft_mod_ai.task_template_catalog import ROOT, load_template


def count_reply(count=0):
    return {"count": count}


def drive(monkeypatch, replies):
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return deepcopy(next(replies))

    monkeypatch.setattr(bounded, "generate_fixed_template_value", generate)
    return calls


def test_record_roundtrip_uses_cardinality_then_exact_record(monkeypatch):
    record = {"trigger": "right click", "owner": "server player"}
    calls = drive(monkeypatch, iter([count_reply(1), record]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={"criterion": "activate"},
        allowed_refs=set(),
    )
    assert result["records"] == [record]
    assert len(calls) == 2
    count_schema = calls[0]["response_schema"]
    assert count_schema["properties"]["count"] == {"type": "integer", "minimum": 0}
    assert "maximum" not in count_schema["properties"]["count"]
    assert set(count_schema["properties"]) == {"count"}
    assert calls[1]["response_schema"] == load_template(
        "feature/behavior_contract/entry_conditions"
    )["record_schema"]


def test_empty_result_is_host_owned_completion(monkeypatch):
    calls = drive(monkeypatch, iter([count_reply(0)]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs=set(),
    )
    assert result["records"] == []
    assert result["reason"]
    assert len(calls) == 1


def test_repeated_record_fails_closed_without_model_continuation(monkeypatch):
    record = {"trigger": "right click", "owner": "server player"}
    drive(monkeypatch, iter([count_reply(2), record, record]))
    with pytest.raises(runner.TemplateBlocked, match="NO_PROGRESS"):
        runner.run_record_template(
            None,
            "feature/behavior_contract/entry_conditions",
            context={},
            allowed_refs=set(),
        )


def test_nonblocking_cardinality_never_delegates_missing_fact_policy(monkeypatch):
    calls = drive(monkeypatch, iter([count_reply(0)]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs=set(),
    )
    assert result["records"] == []
    schema = calls[0]["response_schema"]
    assert set(schema["properties"]) == {"count"}
    assert schema["required"] == ["count"]


def test_catalog_manifests_resolve_every_declared_task():
    for path in ROOT.rglob("*.yaml"):
        task = load_template(path.relative_to(ROOT).with_suffix("").as_posix())
        for identifier in task.get("steps", []):
            assert load_template(identifier)["id"] == identifier


def test_allowed_evidence_is_not_automatically_attached(monkeypatch):
    record = {"trigger": "click", "owner": "server"}
    drive(monkeypatch, iter([count_reply(1), record]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs={"unrelated_a", "unrelated_b"},
    )
    assert result["evidence_refs"] == []


def test_host_context_can_admit_known_evidence(monkeypatch):
    record = {"trigger": "click", "owner": "server"}
    drive(monkeypatch, iter([count_reply(1), record]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={"evidence": ["e1", "not-allowed"]},
        allowed_refs={"e1"},
    )
    assert result["evidence_refs"] == ["e1"]


def test_entry_condition_cardinality_contract_is_host_owned():
    schema = runner.record_response_schema(
        load_template("feature/behavior_contract/entry_conditions")
    )
    assert "status" not in schema["properties"]
    assert "records" not in schema["properties"]
    assert set(schema["properties"]) == {"count"}
