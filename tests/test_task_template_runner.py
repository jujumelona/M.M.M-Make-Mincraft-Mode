from copy import deepcopy

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai import task_template_runner as runner
from minecraft_mod_ai.task_template_catalog import ROOT, load_template


def drive(monkeypatch, replies):
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return deepcopy(next(replies))

    monkeypatch.setattr(bounded, "generate_fixed_template_value", generate)
    return calls


def test_record_roundtrip_uses_one_semantic_record_set(monkeypatch):
    record = {"trigger": "right click", "owner": "server player"}
    calls = drive(monkeypatch, iter([{"records": [record]}]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={"criterion": "activate"},
        allowed_refs=set(),
    )
    assert result["records"] == [record]
    assert len(calls) == 1
    schema = calls[0]["response_schema"]
    assert schema["required"] == ["records"]
    assert set(schema["properties"]) == {"records"}
    assert "maxItems" not in schema["properties"]["records"]


def test_empty_result_is_host_owned_data_not_completion_protocol(monkeypatch):
    calls = drive(monkeypatch, iter([{"records": []}]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs=set(),
    )
    assert result["records"] == []
    assert result["reason"]
    assert len(calls) == 1


def test_repeated_records_are_deduplicated_without_fatal_loop_gate(monkeypatch):
    record = {"trigger": "right click", "owner": "server player"}
    drive(monkeypatch, iter([{"records": [record, record]}]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs=set(),
    )
    assert result["records"] == [record]


def test_record_set_schema_has_no_model_loop_control(monkeypatch):
    calls = drive(monkeypatch, iter([{"records": []}]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs=set(),
    )
    assert result["records"] == []
    schema = calls[0]["response_schema"]
    assert set(schema["properties"]) == {"records"}
    assert schema["required"] == ["records"]
    for forbidden in ("count", "done", "next_work", "blocked_reason", "continuation"):
        assert forbidden not in schema["properties"]


def test_catalog_manifests_resolve_every_declared_task():
    for path in ROOT.rglob("*.yaml"):
        task = load_template(path.relative_to(ROOT).with_suffix("").as_posix())
        for identifier in task.get("steps", []):
            assert load_template(identifier)["id"] == identifier


def test_allowed_evidence_is_not_automatically_attached(monkeypatch):
    record = {"trigger": "click", "owner": "server"}
    drive(monkeypatch, iter([{"records": [record]}]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={},
        allowed_refs={"unrelated_a", "unrelated_b"},
    )
    assert result["evidence_refs"] == []


def test_host_context_can_admit_known_evidence(monkeypatch):
    record = {"trigger": "click", "owner": "server"}
    drive(monkeypatch, iter([{"records": [record]}]))
    result = runner.run_record_template(
        None,
        "feature/behavior_contract/entry_conditions",
        context={"evidence": ["e1", "not-allowed"]},
        allowed_refs={"e1"},
    )
    assert result["evidence_refs"] == ["e1"]


def test_entry_condition_record_contract_is_data_only():
    schema = runner.record_response_schema(
        load_template("feature/behavior_contract/entry_conditions")
    )
    assert schema["required"] == ["records"]
    assert set(schema["properties"]) == {"records"}
    item = schema["properties"]["records"]["items"]
    assert item == load_template("feature/behavior_contract/entry_conditions")["record_schema"]
