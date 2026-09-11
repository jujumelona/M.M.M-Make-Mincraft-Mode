from copy import deepcopy

import pytest

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai import task_template_runner as runner
from minecraft_mod_ai.task_template_catalog import ROOT, load_template


def reply(records=None, blocked_reason="", refs=None):
    return {
        "records": records or [],
        "blocked_reason": blocked_reason,
        "evidence_refs": refs or [],
    }


def drive(monkeypatch, replies):
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return deepcopy(next(replies))

    monkeypatch.setattr(bounded, "generate_fixed_template_value", generate)
    return calls


def test_record_roundtrip_uses_exact_catalog_contract(monkeypatch):
    record = {"trigger": "right click", "owner": "server player"}
    calls = drive(monkeypatch, iter([reply([record])]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={"criterion": "activate"},
        allowed_refs=set(),
    )
    assert result["records"] == [record]
    assert len(calls) == 1
    assert (
        calls[0]["response_schema"]["properties"]["records"]["items"]
        == load_template("feature/behavior_contract/entry_conditions")["record_schema"]
    )


def test_empty_result_is_host_owned_completion(monkeypatch):
    drive(monkeypatch, iter([reply()]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs=set(),
    )
    assert result["records"] == []
    assert result["reason"]


def test_unproven_evidence_is_rejected(monkeypatch):
    record = {"trigger": "x", "owner": "y"}
    drive(monkeypatch, iter([reply([record], refs=["invented"])]))
    with pytest.raises(ValueError, match="TEMPLATE_EVIDENCE"):
        runner.run_record_template(
            None,
            "feature/behavior_contract/entry_conditions",
            context={},
            allowed_refs=set(),
        )


def test_repeated_record_stops_without_model_continuation(monkeypatch):
    record = {"trigger": "right click", "owner": "server player"}
    drive(monkeypatch, iter([reply([record, record])]))
    with pytest.raises(runner.TemplateBlocked, match="NO_PROGRESS"):
        runner.run_record_template(
            None,
            "feature/behavior_contract/entry_conditions",
            context={},
            allowed_refs=set(),
        )


def test_missing_information_blocks_without_status_protocol(monkeypatch):
    drive(monkeypatch, iter([reply(blocked_reason="trigger not established")]))
    with pytest.raises(runner.TemplateBlocked, match="trigger not established"):
        runner.run_record_template(
            None,
            "feature/behavior_contract/entry_conditions",
            context={},
            allowed_refs=set(),
        )


def test_catalog_manifests_resolve_every_declared_task():
    for path in ROOT.rglob("*.yaml"):
        task = load_template(path.relative_to(ROOT).with_suffix("").as_posix())
        for identifier in task.get("steps", []):
            assert load_template(identifier)["id"] == identifier


def test_allowed_evidence_is_not_automatically_attached(monkeypatch):
    record = {"trigger": "click", "owner": "server"}
    drive(monkeypatch, iter([reply([record])]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs={"unrelated_a", "unrelated_b"},
    )
    assert result["evidence_refs"] == []


def test_legacy_status_protocol_is_absent():
    schema = runner.record_response_schema(
        load_template("feature/behavior_contract/entry_conditions")
    )
    assert "status" not in schema["properties"]
    assert "records" in schema["properties"]
