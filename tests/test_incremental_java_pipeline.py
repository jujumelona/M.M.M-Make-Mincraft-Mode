"""The actual production entrypoint uses operation authoring and durable AST repair."""
from __future__ import annotations

import json
from incremental_value_model import structured_values

from test_incremental_java_author import OperationCoder, _inputs, _compile_claims


def production_fixture(tmp_path, monkeypatch):
    import minecraft_mod_ai.construction_host_source as host
    behavior, spec, context, messages = _inputs()
    prompt = json.loads(messages[-1]["content"])
    prompt.update(inspected_target_api={}, materialized_prerequisite_api=[], host_state_work_order_scope={})
    events = prompt["available_fabric_callbacks"]
    context.update(project_root=str(tmp_path), java_classpath=["fixture"],
                   target={"loader": "fabric", "java_version": 25}, prerequisite_hashes={})
    context.pop("host_direct_feature_source")

    def forbidden(*args, **kwargs):
        raise AssertionError("New implementation must not enter capability/proof/whole-member planning")

    for target in ("feature_execution_route.select_feature_execution_route",
                   "feature_source_work_graph.build_source_work_graph",
                   "source_owned_site_choice.choose_verified_source_site",
                   "runtime_mutation_author.verify_native_runtime_mutation_coverage",
                   "feature_registration_proof.validate_registered_event_sources",
                   "first_pass_java_source.compose_first_pass_java"):
        monkeypatch.setattr("minecraft_mod_ai." + target, forbidden)
    monkeypatch.setattr(host, "_build_host_java_prompt", lambda *args, **kwargs: (
        {}, behavior["intent"], [behavior], prompt, []))
    monkeypatch.setattr("minecraft_mod_ai.host_fabric_callback_abi.callback_interfaces", lambda *args, **kwargs: events)
    def verify(source, **kwargs):
        try:
            _compile_claims(tmp_path, source, execute=False)
        except AssertionError as exc:
            raise RuntimeError(str(exc)) from exc
    monkeypatch.setattr("minecraft_mod_ai.construction_project_compile.verify_java_source_with_dependencies", verify)
    monkeypatch.setattr("minecraft_mod_ai.java_source_snapshot.record_compiled_sources", lambda *args, **kwargs: None)
    def build(coder):
        return host.compose_host_java_source(coder, spec, context, package="example", name="Feature",
                                            owner="example.Feature", behaviors=[behavior])
    return build


def test_production_uses_incremental_author_and_resumes_without_inference(tmp_path, monkeypatch):
    build = production_fixture(tmp_path, monkeypatch)
    coder = OperationCoder()
    source = build(coder)
    count = len(coder.requests) + len(coder.expression_requests)
    assert "mmmField1" in source
    assert build(coder) == source
    assert count == len(coder.requests) + len(coder.expression_requests)
    _compile_claims(tmp_path, source)


def test_three_independent_operands_can_be_fixed_without_restarting_the_feature(tmp_path, monkeypatch):
    build = production_fixture(tmp_path, monkeypatch)
    import minecraft_mod_ai.construction_host_source as host
    from minecraft_mod_ai.construction_direct_writer import _read_checkpoint
    original_save = host._save_checkpoint
    saved_programs = []
    def save(path, identity, records):
        original_save(path, identity, records)
        if "program_json" in records:
            saved_programs.append(json.loads(_read_checkpoint(path, identity)["program_json"]))
    monkeypatch.setattr(host, "_save_checkpoint", save)
    @structured_values
    class FixableCoder(OperationCoder):
        def __init__(self):
            super().__init__()
            self.repaired = []

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("repair_target"):
                return {"repair": "mapped_operand"}
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

        def value_answer(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            if "javac_diagnostic" in task:
                self.repaired.append((task["current_member"]["id"], task["repair_slot"]))
                return super().value_answer(role, messages, data_name=data_name, source_frame=source_frame)
            bad = {"initializer": '"missingInitial"', "condition": "1", "value": '"missingAmount"'}
            if task["expression_slot"] in bad:
                return source_frame["prefix"] + bad[task["expression_slot"]] + source_frame["suffix"]
            return super().value_answer(role, messages, data_name=data_name, source_frame=source_frame)

    coder = FixableCoder()
    source = build(coder)
    assert len(coder.repaired) == 3
    assert len(set(coder.repaired)) == 3
    _compile_claims(tmp_path, source)
    assert len(saved_programs) == 3
    assert all(program["schema"] == "mmm/incremental-java-v1" for program in saved_programs)
    assert all(len(program["members"]) == 3 for program in saved_programs)


def test_an_empty_registrar_is_continued_with_new_code_instead_of_an_unmapped_failure(tmp_path, monkeypatch):
    build = production_fixture(tmp_path, monkeypatch)
    @structured_values
    class PrematureCoder(OperationCoder):
        premature = True

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if self.premature and data_name.endswith("_kind"):
                self.premature = False
                return {"op": "complete"}
            if not self.requests:
                assert "CONSTRUCTION_FILE_BEHAVIOR_METHOD_MISSING" in task["javac_diagnostic"]
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)
    _compile_claims(tmp_path, build(PrematureCoder()))


def test_interrupted_compiler_repair_resumes_its_saved_ast_without_reauthoring(tmp_path, monkeypatch):
    import pytest
    build = production_fixture(tmp_path, monkeypatch)
    @structured_values
    class InterruptedCoder(OperationCoder):
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("repair_target"):
                raise RuntimeError("model worker interrupted")
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

        def value_answer(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            if task["expression_slot"] == "initializer":
                return source_frame["prefix"] + '"missingInitial"' + source_frame["suffix"]
            return super().value_answer(role, messages, data_name=data_name, source_frame=source_frame)
    with pytest.raises(RuntimeError, match="model worker interrupted"):
        build(InterruptedCoder())

    @structured_values
    class ResumedCoder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert data_name.endswith("repair_target")
            return {"repair": "mapped_operand"}

        def value_answer(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            assert task["current_member"]["kind"] == "field"
            assert "missingInitial" in task["javac_diagnostic"]
            return source_frame["prefix"] + "0" + source_frame["suffix"]
    _compile_claims(tmp_path, build(ResumedCoder()))
