"""Regressions from the 1852ca54 Colab trace; Java runs are local, not Fabric E2E."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai.atomic_concern_source import _state_default_literal
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.custom_module_generator import _call_atomic_java_region
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema


def _compile_run(tmp_path, members, probe):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("Java compiler/runtime unavailable")
    target = tmp_path / "Probe.java"
    target.write_text("public class Probe {\n" + members +
                      "\npublic static void main(String[] args) {\n" + probe + "\n}}", encoding="utf-8")
    result = subprocess.run([javac, str(target)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([java, "-cp", str(tmp_path), "Probe"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind,value,expected", [
    ("float", "1.0", "1.0f"), ("float", "1e-3", "0.001f"),
    ("long", "3000000000", "3000000000L"),
])
def test_host_numeric_defaults_compile_without_model_repair(tmp_path, kind, value, expected):
    members = f"static {kind} value = {_state_default_literal(kind, value)};"
    _compile_run(tmp_path, members,
                 f'if (value != {expected}) throw new AssertionError("default changed");')


class ScalarRouter:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_tool_decision(self, role, messages, **kwargs):
        schema = kwargs["parameters"]
        # The failing request asked for fields + methods + static_initializers
        # as nested JSON arrays. No request in the replacement may do that.
        assert_atomic_model_schema(schema, surface="first-pass Java")
        assert all(
            value.get("type") not in {"array", "object"}
            for value in schema["properties"].values()
            if isinstance(value.get("type"), str)
        )
        self.calls.append((messages, kwargs))
        value = next(self.responses)
        Draft202012Validator(schema).validate(value)
        return value


def _messages():
    return [{"role": "user", "content": json.dumps({
        "response_region": "members", "host_selected_class": "Probe",
        "concern": {"name": "initialization"},
        "available_sibling_api": [{"owner_concern": "variables", "kind": "field",
            "symbol": "integrity", "declaration": "private static float integrity = 1.0f;"}],
    })}]


def test_structured_assembly_receives_first_pass_compiler_contract() -> None:
    router = ScalarRouter([{"part": "done"}])
    messages = [{"role": "user", "content": json.dumps({
        "response_region": "members",
        "host_selected_class": "Probe",
        "concern": {"name": "concurrency_hazards"},
        "generation_recipe": {
            "first_pass_goal": "compile on the first generated candidate",
            "compiler_first_rules": [
                "Never reassign a final field.",
                "Narrow Object with an explicit runtime type check.",
            ],
            "jdk_package_anchors": {
                "locks": "java.util.concurrent.locks",
            },
            "sibling_api_is_authoritative": True,
            "never_mutate_final_sibling_fields": True,
        },
    })}]

    _call_atomic_java_region(router, messages, output_token_ceiling=512)

    request_messages, _kwargs = router.calls[0]
    system = request_messages[0]["content"]
    payload = json.loads(request_messages[-1]["content"])

    assert payload["compiler_contract"]["first_pass_goal"] == (
        "compile on the first generated candidate"
    )
    assert payload["compiler_contract"]["rules"] == [
        "Never reassign a final field.",
        "Narrow Object with an explicit runtime type check.",
    ]
    assert payload["compiler_contract"]["jdk_package_anchors"]["locks"] == (
        "java.util.concurrent.locks"
    )
    assert payload["compiler_contract"]["sibling_api_is_authoritative"] is True
    assert payload["compiler_contract"]["never_mutate_final_sibling_fields"] is True
    assert "first-pass compilable Java" in system
    assert "Lock/ReentrantLock live in java.util.concurrent.locks" in system


def test_java_modifiers_are_never_model_authored() -> None:
    captured: list[tuple[list[object], set[str]]] = []

    class Router:
        def __init__(self):
            self.field_emitted = False

        def generate_tool_decision(self, role, messages, **kwargs):
            del role
            schema = kwargs["parameters"]
            properties = schema["properties"]
            payload = json.loads(messages[-1]["content"])
            captured.append((payload["assembly"]["path"], set(properties)))

            if set(properties) == {"part"}:
                choices = properties["part"]["enum"]
                if "fields" in choices and not self.field_emitted:
                    self.field_emitted = True
                    return {"part": "fields"}
                return {"part": "done"}

            assert set(properties) == {"type", "name", "initializer"}
            return {"type": "int", "name": "player", "initializer": "1"}

    rendered = _call_atomic_java_region(
        Router(),
        _messages(),
        output_token_ceiling=512,
    )

    assert rendered == "private static int player = 1;"
    assert all("modifiers" not in path for path, _properties in captured)
    assert all(
        "modifiers" not in properties
        and "visibility" not in properties
        and not any(key.startswith("is_") for key in properties)
        for _path, properties in captured
    )


def test_outer_public_api_modifiers_come_from_host_contract() -> None:
    captured_paths: list[list[object]] = []

    class Router:
        def __init__(self):
            self.method_emitted = False
            self.body_emitted = False

        def generate_tool_decision(self, role, messages, **kwargs):
            del role
            schema = kwargs["parameters"]
            properties = schema["properties"]
            payload = json.loads(messages[-1]["content"])
            captured_paths.append(payload["assembly"]["path"])

            if set(properties) == {"part"}:
                choices = properties["part"]["enum"]
                if "methods" in choices and not self.method_emitted:
                    self.method_emitted = True
                    return {"part": "methods"}
                if "body" in choices and not self.body_emitted:
                    self.body_emitted = True
                    return {"part": "body"}
                return {"part": "done"}
            if set(properties) == {"return_type", "name"}:
                return {"return_type": "void", "name": "launch"}
            if set(properties) == {"value"}:
                return {"value": "return;"}
            raise AssertionError(f"unexpected schema properties: {sorted(properties)}")

    messages = [{
        "role": "user",
        "content": json.dumps({
            "response_region": "members",
            "host_selected_class": "Probe",
            "concern": {"name": "initialization"},
            "host_grounding": {
                "implementation_contract": {
                    "symbol": "Probe",
                    "public_api": ["public static synchronized void launch()"],
                },
            },
        }),
    }]

    rendered = _call_atomic_java_region(
        Router(),
        messages,
        output_token_ceiling=512,
    )

    assert rendered == "public static synchronized void launch() {\n    return;\n}"
    assert all("modifiers" not in path for path in captured_paths)


def test_initialization_is_assembled_from_native_scalars_and_executes(tmp_path):
    router = ScalarRouter([
        {"part": "methods"}, {"return_type": "void", "name": "initPlayerState"},
        {"part": "body"}, {"value": "integrity = 1.0f;"},
        {"part": "done"}, {"part": "done"},
    ])
    members = _call_atomic_java_region(router, _messages(), output_token_ceiling=512)
    assert len(router.calls) == 6
    assert all("repair" not in kwargs["tool_name"] for _, kwargs in router.calls)
    _compile_run(tmp_path, "static float integrity;\n" + members,
                 'initPlayerState(); if (integrity != 1.0f) throw new AssertionError();')


def test_modified_type_cannot_reach_renderer_or_trigger_region_retry():
    router = ScalarRouter([
        {"part": "fields"}, {"type": "private static float", "name": "localValue"},
    ])
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_JAVA_ASSEMBLY_INVALID"):
        _call_atomic_java_region(router, _messages(), output_token_ceiling=None)
    assert len(router.calls) == 2


def test_stringified_methods_cannot_reenter_whole_region_generation():
    router = ScalarRouter([{"methods": '[{"name":"initPlayerState"}]'}])
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_JAVA_ASSEMBLY_INVALID"):
        _call_atomic_java_region(router, _messages(), output_token_ceiling=None)
    assert len(router.calls) == 1


def test_sibling_state_name_is_excluded_from_new_field_schema():
    router = ScalarRouter([{"part": "fields"}, {"type": "float", "name": "integrity"}])
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_JAVA_ASSEMBLY_INVALID"):
        _call_atomic_java_region(router, _messages(), output_token_ceiling=None)
    assert len(router.calls) == 2


def test_model_numeric_initializers_use_the_same_typed_renderer(tmp_path):
    router = ScalarRouter([
        {"part": "fields"}, {"type": "float", "name": "localValue", "initializer": "1.0"},
        {"part": "done"}, {"part": "done"},
    ])
    members = _call_atomic_java_region(router, _messages(), output_token_ceiling=None)
    _compile_run(tmp_path, members, 'if (localValue != 1.0f) throw new AssertionError();')


def test_unbalanced_body_is_terminal_before_executor_can_retry():
    router = ScalarRouter([
        {"part": "methods"}, {"return_type": "void", "name": "bad"},
        {"part": "body"}, {"value": "if (true) {"},
        {"part": "done"}, {"part": "done"},
    ])
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_JAVA_ASSEMBLY_INVALID"):
        _call_atomic_java_region(router, _messages(), output_token_ceiling=None)
    assert len(router.calls) == 6


def test_public_api_check_accepts_multiline_host_declarations():
    from minecraft_mod_ai.implementation_graph_execution import public_api_errors

    contract = {"public_api": ["public static synchronized void initializeState(String trigger, java.util.Map<String, Object> context)"]}
    source = "public static synchronized void initializeState(\nString trigger,\njava.util.Map<String, Object> context\n) {}"
    assert public_api_errors(source, contract) == ()
    assert public_api_errors(source.replace("String trigger", "int trigger"), contract)


def test_compact_state_design_does_not_fall_back_to_free_form_atomic_coder(tmp_path):
    from minecraft_mod_ai.atomic_concern_source import AtomicConcernExecutor
    from minecraft_mod_ai.implementation_graph_execution import (
        _bind_atomic_leaf_contract,
    )

    requirements = {
        "R1": "## state_model",
        "R2": "- variables: PlayerCredit (PlayerCredit, Owner=Global, Type=Integer, Default=0, Domain=-999~999M), "
              "ShipIntegrity (ShipIntegrity, Owner=Local, Type=Float, Default=1.0, Domain=0.0~1.0)",
        "R3": "- initialization: reset ShipIntegrity to 1.0 and expose its value",
    }
    task = {"task_id": "logged-compact-state"}
    node = {"symbol": "AuthoredStateModel", "obligations": [json.dumps({
        "instruction": json.dumps({"concern": concern, "section": "state_model"}),
        "source_requirements": requirements,
    }) for concern in ("variables", "initialization")]}
    section, concerns = _bind_atomic_leaf_contract(task, node, requirements)
    compile_calls = []

    executor = AtomicConcernExecutor(
        root=tmp_path,
        target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java",
        symbol="AuthoredStateModel",
        original="public final class AuthoredStateModel {\n// MMM_AUTHORED_FEATURE_BODY\n}\n",
        task=task,
        section=section,
        concerns=concerns,
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=lambda _messages: pytest.fail(
            "state_model must not fall back to the free-form atomic coder"
        ),
        compile_java=lambda root: compile_calls.append(root),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: pytest.fail(
            "state_model contract failure must happen before source mutation"
        ),
    )

    with pytest.raises(
        CustomModuleGenerationError,
        match="STRUCTURED_STATE_CONTRACT_REQUIRED",
    ):
        executor.run()

    assert compile_calls == []
