"""Integration proof for host-assembled Java route and complete-file checkpoint."""
from __future__ import annotations

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
