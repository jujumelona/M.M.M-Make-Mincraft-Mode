"""Integration proof for host-assembled Java route and complete-file checkpoint."""
from __future__ import annotations

import json
import pytest

from minecraft_mod_ai.construction_host_source import compose_host_java_source


SPEC = {
    "language": "java",
    "target_path": "src/main/java/example/Bridge.java",
    "requirement": "Implement a Java readiness method.",
    "bindings": {"package_name": "example", "class_name": "Bridge"},
}


class Router:
    def __init__(self, *, forbidden=False):
        self.calls = []
        self.forbidden = forbidden

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        if self.forbidden:
            raise AssertionError("Accepted compilation unit must be checkpointed")
        self.calls.append(data_name)
        field = next(iter(parameters["properties"]))
        if data_name.endswith("_count"):
            return {"count": 1 if "units" in data_name else 0}
        if field == "superclass":
            return {"superclass": ""}
        if field == "interfaces":
            return {"interfaces": ""}
        if data_name.endswith("_value_kind"):
            return {"kind": "method"}
        if data_name.endswith("_value_name"):
            return {"name": "ready"}
        if data_name.endswith("_value_purpose"):
            return {"purpose": "Return readiness"}
        if data_name == "submit_java_declaration_1":
            return {
                "visibility": "public", "static": False,
                "return_type": "boolean", "type_parameters": "",
                "parameters": [], "throws": [], "annotations": [],
            }
        raise AssertionError(data_name)

    def generate_java_source(self, role, messages, *, data_name, source_frame):
        if self.forbidden:
            raise AssertionError("Accepted compilation unit must be checkpointed")
        self.calls.append(data_name)
        return source_frame["prefix"] + "return true;" + source_frame["suffix"]


def _context(tmp_path):
    return {
        "target": {"loader": "fabric", "minecraft": "26.2", "java_version": 25},
        "project_root": str(tmp_path),
        "prerequisite_hashes": {}, "java_classpath": ["/fake/classpath"],
        "symbols": {}, "source_pages": {},
        "authoring_contract": {}, "binding_pages": {},
    }


def test_compilation_unit_is_host_composed_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "host-source-test-fingerprint")
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._evidence",
        lambda *args, **kwargs: {"class_members": []})
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._read_materialized_context",
        lambda *args, **kwargs: [])
    compiler_calls = []
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._verify_full_source",
        lambda source, *args, **kwargs: compiler_calls.append(source))
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_coverage.record_authored_source_calls",
        lambda *args, **kwargs: None)
    first = Router()
    output = compose_host_java_source(first, SPEC, _context(tmp_path),
        package="example", name="Bridge", owner="example.Bridge")
    assert output.startswith("package example;\n\npublic class Bridge {")
    assert "public boolean ready()" in output
    assert "return true;" in output
    assert not any("imports" in item for item in first.calls)
    assert len(compiler_calls) == 1
    second = Router(forbidden=True)
    assert compose_host_java_source(second, SPEC, _context(tmp_path),
        package="example", name="Bridge", owner="example.Bridge") == output
    assert len(compiler_calls) == 2


def test_failed_compiler_candidate_resumes_at_member_without_replaying_rejected_repair(tmp_path, monkeypatch):
    from minecraft_mod_ai.construction_journal import JournaledConstructionRouter

    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "host-source-test-fingerprint")
    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._evidence",
                        lambda *args, **kwargs: {"class_members": []})
    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._read_materialized_context",
                        lambda *args, **kwargs: [])
    monkeypatch.setattr("minecraft_mod_ai.construction_coverage.record_authored_source_calls",
                        lambda *args, **kwargs: None)

    def compiler(source, *args, **kwargs):
        if "return true;" not in source:
            line = next(i for i, text in enumerate(source.splitlines(), 1) if "return " in text)
            raise RuntimeError(f"Bridge.java:{line}: error: cannot find symbol")

    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._verify_full_source", compiler)
    context = {**_context(tmp_path), "feature_work_order": {
        "acceptance": {"max_compiler_repair_attempts": 1}}}

    class BrokenRouter(Router):
        def generate_java_source(self, role, messages, *, data_name, source_frame):
            self.calls.append(data_name)
            body = "return missing;" if data_name.startswith("submit_") else "return stillMissing;"
            return source_frame["prefix"] + body + source_frame["suffix"]

    first = BrokenRouter()
    with pytest.raises(RuntimeError, match="cannot find symbol"):
        compose_host_java_source(JournaledConstructionRouter(first, tmp_path, SPEC, context),
            SPEC, context, package="example", name="Bridge", owner="example.Bridge")
    assert not list((tmp_path / ".mmm/construction_host_files").glob("*.json"))

    class RepairOnlyRouter(Router):
        def generate_structured_data(self, *args, **kwargs):
            raise AssertionError("Do not regenerate the plan or unaffected members")

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            assert data_name.startswith("repair_compiler_member_")
            self.calls.append(data_name)
            task = json.loads(messages[-1]["content"])
            assert "return stillMissing;" in task["current_member"]
            assert "cannot find symbol" in task["javac_diagnostic"]
            return source_frame["prefix"] + "return true;" + source_frame["suffix"]

    second = RepairOnlyRouter()
    output = compose_host_java_source(
        JournaledConstructionRouter(second, tmp_path, SPEC, context),
        SPEC, context, package="example", name="Bridge", owner="example.Bridge")
    assert "return true;" in output
    assert len(second.calls) == 1
    assert not list((tmp_path / ".mmm/construction_host_candidates").glob("*.json"))



def test_callback_api_evidence_uses_actual_trigger_not_only_feature_title(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "host-callback-trigger-evidence-test")
    captured = []
    def evidence(_context, intent, **_kwargs):
        captured.append(intent)
        return {"class_members": []}
    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._evidence", evidence)
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._read_materialized_context",
        lambda *a, **kw: [])
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._verify_full_source",
        lambda *a, **kw: None)
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_coverage.record_authored_source_calls",
        lambda *a, **kw: None)
    compose_host_java_source(
        Router(), SPEC, _context(tmp_path),
        package="example", name="Bridge", owner="example.Bridge",
        behaviors=[{
            "behavior_id": "ship.purchase",
            "intent": "Upgrade ship",
            "trigger": "Player opens trading interface and selects purchase",
            "effects": "Deduct credits and upgrade ship weapons",
        }],
    )
    assert len(captured) == 1
    assert "Player opens trading interface" in captured[0]
    assert "Deduct credits and upgrade ship weapons" in captured[0]
    assert "Upgrade ship" in captured[0]
