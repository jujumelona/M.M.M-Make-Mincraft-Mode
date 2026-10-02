"""Production native response -> component admission -> real Java execution."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai.atomic_concern_source import AtomicConcernExecutor
from minecraft_mod_ai.custom_module_errors import (
    AtomicJavaDecisionError,
)
from minecraft_mod_ai.custom_module_generator import (
    _call_atomic_java_region,
)
from minecraft_mod_ai.model_adapters.base import NativeToolDecisionRejected


def messages():
    return [{"role": "user", "content": json.dumps({
        "response_region": "members", "host_selected_class": "AuthoredStateModel",
        "concern": {"name": "variables"},
    })}]


class Router:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_tool_decision(self, role, request, **kwargs):
        self.calls.append((deepcopy(request), deepcopy(kwargs)))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


def executor(tmp_path, call):
    return AtomicConcernExecutor(
        root=tmp_path, target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java", symbol="AuthoredStateModel",
        original="// MMM_AUTHORED_FEATURE_BODY\n", task={"task_id": "test"},
        section="state_model", concerns=({"sequence": 0, "identifier": "variables",
            "concern": "variables", "task": "state", "rules": []},),
        grounding={}, dependency_source="", require_initialize=False, call_coder=call,
        compile_java=lambda _: SimpleNamespace(status="PASS"), compile_log=lambda _: "",
        write_source=lambda *_: None,
    )


def test_host_name_is_reserved_before_any_nested_body_is_generated():
    router = Router([{"part": "classes"}, {"name": "AuthoredStateModel"}])
    with pytest.raises(AtomicJavaDecisionError, match="ATOMIC_JAVA_ASSEMBLY_INVALID") as error:
        _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert len(router.calls) == 2
    schema = router.calls[-1][1]["parameters"]
    assert not Draft202012Validator(schema).is_valid({"name": "AuthoredStateModel"})
    assert "AuthoredStateModel" in error.value.response_text


@pytest.mark.parametrize("category,item", [
    ("fields", {"name": "credits", "type": 5}),
    ("methods", {"name": "value", "return_type": False}),
])
def test_invalid_declaration_stops_without_replacing_accepted_sibling(category, item):
    router = Router([
        {"part": "fields"}, {"name": "other", "type": "int", "initializer": "42"},
        {"part": category}, item,
    ])
    with pytest.raises(AtomicJavaDecisionError, match="ATOMIC_JAVA_ASSEMBLY_INVALID"):
        _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert len(router.calls) == 4
    accepted = json.loads(router.calls[-1][0][-1]["content"])["assembly"]["accepted_structure"]
    assert accepted["fields"][0] == {"name": "other", "type": "int", "initializer": "42"}


def test_cross_category_type_collision_stops_before_second_body():
    router = Router([
        {"part": "records"}, {"name": "Data"}, {"part": "done"},
        {"part": "classes"}, {"name": "Data"},
    ])
    with pytest.raises(AtomicJavaDecisionError, match="ATOMIC_JAVA_ASSEMBLY_INVALID"):
        _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert len(router.calls) == 5


def test_recursive_type_and_parent_identity_survive_native_assembly():
    router = Router([
        {"part": "classes"}, {"name": "ShipData"},
        {"part": "fields"}, {"type": "ShipData", "name": "next"}, {"part": "done"},
        {"part": "done"}, {"part": "done"},
    ])
    source = _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert "ShipData next;" in source
    parent = json.loads(router.calls[3][0][-1]["content"])["assembly"]
    assert parent["path"] == ["classes", 0, "fields", 0]
    assert parent["accepted_structure"]["classes"][0]["name"] == "ShipData"


@pytest.mark.parametrize("name", [[], {}, None, 12])
def test_invalid_name_stops_before_class_body(name):
    router = Router([{"part": "classes"}, {"name": name}])
    with pytest.raises(AtomicJavaDecisionError, match="ATOMIC_JAVA_ASSEMBLY_INVALID"):
        _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert len(router.calls) == 2


def test_assembly_requires_explicit_completion_and_has_finite_budget(monkeypatch):
    from minecraft_mod_ai import atomic_java_assembly as assembly
    from minecraft_mod_ai.implementation_ir import OutputBudgetExhausted
    monkeypatch.setattr(assembly, "MAX_ASSEMBLY_CALLS", 3)
    router = Router([
        {"part": "fields"}, {"name": "x", "type": "int"}, {"part": "fields"},
    ])
    with pytest.raises(OutputBudgetExhausted, match="OUTPUT_BUDGET_EXHAUSTED"):
        _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert len(router.calls) == 3


def test_native_rejection_is_terminal_and_preserves_transport_evidence():
    rejection = {"original_tool": "emit_java_part", "failure_code": "TOOL_SCHEMA_INVALID",
                 "raw_arguments": '{"methods": "invalid-array"}'}
    router = Router([NativeToolDecisionRejected("emit_java_part", [rejection])])
    with pytest.raises(AtomicJavaDecisionError, match="ATOMIC_JAVA_ASSEMBLY_INVALID") as error:
        _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert len(router.calls) == 1
    assert "invalid-array" in error.value.response_text


def test_null_response_and_missing_response_have_different_evidence():
    missing = AtomicJavaDecisionError("missing")
    null = AtomicJavaDecisionError("invalid null", response=None)
    assert missing.response_sha256 is None
    assert null.response_text == "null"
    assert null.response_sha256 == hashlib.sha256(b"null").hexdigest()


def test_completed_bodies_leave_context_but_current_component_stays_exact():
    from minecraft_mod_ai.atomic_java_assembly import _assembly_context

    root = {"methods": [
        {"name": "first", "return_type": "void", "body": ["runPrevious();"] * 27},
        {"name": "second", "return_type": "void", "body": ["runCurrent();"]},
    ]}
    snapshot = _assembly_context(root, ["methods", 1, "body", 1])
    assert "body" not in snapshot["methods"][0]
    assert snapshot["methods"][0]["name"] == "first"
    assert len(snapshot["methods"][0]["accepted_body_sha256"]) == 64
    assert snapshot["methods"][1] == root["methods"][1]
    assert len(root["methods"][0]["body"]) == 27
    closed = _assembly_context(root, [])
    assert all("body" not in method for method in closed["methods"])


def test_context_pressure_returns_to_decomposition_before_inference():
    from types import SimpleNamespace

    from minecraft_mod_ai.atomic_java_assembly import JavaStructureAssembly
    from minecraft_mod_ai.implementation_ir import OutputBudgetExhausted

    router = Router([])
    config = SimpleNamespace(adapter="test", max_context=8192, max_new_tokens=2048)
    assembly = JavaStructureAssembly(router.generate_tool_decision,
                                     {"task_authority": {"requirements": "x" * 16000}},
                                     output_token_ceiling=None, config=config)
    with pytest.raises(OutputBudgetExhausted, match="context needs decomposition"):
        assembly.run({"type": "object", "properties": {
            "fields": {"type": "array", "items": {"type": "object", "properties": {}}},
        }})
    assert not router.calls


@pytest.mark.parametrize("wrapped", [False, True])
def test_exact_runtime_context_overflow_returns_to_decomposition(wrapped):
    from minecraft_mod_ai.implementation_ir import OutputBudgetExhausted
    from minecraft_mod_ai.llama_exact_context import ExactContextOverflow
    from minecraft_mod_ai.model_adapters.base import ModelBackendError

    error = ExactContextOverflow("input exceeds live runtime context")
    if wrapped:
        error = ModelBackendError(role="coder", model_id="test", cause=error)
    router = Router([error])
    with pytest.raises(OutputBudgetExhausted, match="requires decomposition"):
        _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert len(router.calls) == 1


def test_host_owned_compile_failure_does_not_request_model_repair(tmp_path):
    run = executor(tmp_path, lambda _: pytest.fail("host declarations cannot be model-repaired"))
    run.host_owned_concerns.add("variables")
    run.compile_log = lambda _: "AuthoredStateModel.java:3: error: invalid host literal"
    run.source = (
        "public class AuthoredStateModel {\n"
        "// MMM_ATOMIC_CONCERN_VARIABLES_MEMBERS_START\n"
        "static float integrity = 1.0;\n"
        "// MMM_ATOMIC_CONCERN_VARIABLES_MEMBERS_END\n}"
    )
    from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError

    with pytest.raises(CustomModuleGenerationError, match="STRUCTURED_STATE_HOST_COMPILER_INVALID"):
        run._repair_once(SimpleNamespace(status="FAIL"))
