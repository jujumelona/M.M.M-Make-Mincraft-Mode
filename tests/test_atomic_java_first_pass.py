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
        assert all(value.get("type") not in {"array", "object"}
                   for value in schema["properties"].values())
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


def test_compact_design_through_executor_compiles_and_runs_without_repairs(tmp_path):
    from types import SimpleNamespace

    from minecraft_mod_ai.atomic_concern_source import AtomicConcernExecutor
    from minecraft_mod_ai.implementation_graph_execution import (
        _bind_atomic_leaf_contract,
    )

    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("Java compiler/runtime unavailable")
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
    router = ScalarRouter([
        {"part": "methods"}, {"return_type": "void", "name": "reset"},
        {"part": "body"}, {"value": "ShipIntegrity = 1.0f;"}, {"part": "done"},
        {"part": "methods"}, {"return_type": "float", "name": "integrity"},
        {"part": "body"}, {"value": "return ShipIntegrity;"}, {"part": "done"},
        {"part": "done"},
    ])
    target = tmp_path / "AuthoredStateModel.java"
    compilations = []

    def compile_java(_root):
        result = subprocess.run([javac, str(target)], capture_output=True, text=True, check=False)
        compilations.append(result)
        assert result.returncode == 0, result.stderr
        return SimpleNamespace(status="PASS")

    executor = AtomicConcernExecutor(
        root=tmp_path, target=target, relative="AuthoredStateModel.java", symbol="AuthoredStateModel",
        original="public final class AuthoredStateModel {\n// MMM_AUTHORED_FEATURE_BODY\n}\n",
        task=task, section=section, concerns=concerns, grounding={}, dependency_source="",
        require_initialize=False,
        call_coder=lambda messages: _call_atomic_java_region(router, messages, output_token_ceiling=512),
        compile_java=compile_java, compile_log=lambda _: "",
        write_source=lambda path, source: path.write_text(source, encoding="utf-8"),
    )
    result = executor.run()
    assert result["repair_count"] == 0
    assert len(compilations) == 2
    assert result["source"].count("float ShipIntegrity = 1.0f;") == 1
    assert all(json.loads(messages[-1]["content"])["concern"]["name"] == "initialization"
               for messages, _ in router.calls)
    probe = tmp_path / "Probe.java"
    probe.write_text('public class Probe { public static void main(String[] args) { '
                     'AuthoredStateModel.reset(); if (AuthoredStateModel.integrity() != 1.0f) throw new AssertionError(); }}', encoding="utf-8")
    built = subprocess.run([javac, "-cp", str(tmp_path), str(probe)], capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([java, "-cp", str(tmp_path), "Probe"], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stderr
