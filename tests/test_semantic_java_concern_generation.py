from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.atomic_concern_source import _messages, _parse_region_content
from minecraft_mod_ai.custom_module_generator import (
    _call_coder,
    _run_atomic_ir_generation,
)


class DirectTextRouter:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return self.response

    def generate_tool_decision(self, *args, **kwargs):
        raise AssertionError("semantic Java production must not use tool-decision JSON")


def test_concern_prompt_requests_plain_java_not_custom_ast():
    messages = _messages(
        section="algorithm",
        concern={
            "sequence": 0,
            "identifier": "algorithm.transitions",
            "concern": "transitions",
            "task": "implement transitions",
        },
        task={"task_id": "probe", "implementation_obligations": []},
        grounding={},
        dependency_source="",
        current_source=(
            "public final class Probe {\n"
            "// MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_START\n"
            "// MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_END\n"
            "}\n"
        ),
        response_region="members",
        sibling_concerns=(),
        host_symbol="Probe",
    )

    system = messages[0]["content"]
    payload = json.loads(messages[1]["content"])

    assert "Return only compile-ready Java class-body source" in system
    assert "Call emit_java_structure" not in system
    assert payload["scope"]["required_output_format"] == "plain_java_source"
    assert payload["scope"]["model_tools_enabled"] is False
    assert payload["generation_recipe"]["no_json_ast_protocol"] is True


def test_direct_semantic_java_coder_disables_tools():
    router = DirectTextRouter("private static int credits = 0;")

    source = _call_coder(
        router,
        [{"role": "user", "content": "emit selected Java region"}],
        force_non_thinking=True,
        structured_java_region=False,
    )

    assert source == "private static int credits = 0;\n"
    assert len(router.calls) == 1
    role, _messages_arg, kwargs = router.calls[0]
    assert role == "coder"
    assert kwargs["response_format"] == "text"
    assert kwargs["enable_tools"] is False
    assert kwargs["force_non_thinking"] is True


def test_semantic_region_parser_accepts_complete_java_declarations():
    source = _parse_region_content(
        (
            "private static int playerCredits = 0;\n"
            "private static boolean currentPlanetSurface = true;\n"
            "private static void addCredits(int amount) {\n"
            "    playerCredits += amount;\n"
            "}\n"
        ),
        response_region="members",
    )

    assert "playerCredits = 0;" in source
    assert "currentPlanetSurface = true;" in source
    assert "addCredits(int amount)" in source


def test_production_atomic_route_calls_plain_text_coder(monkeypatch, tmp_path):
    import minecraft_mod_ai.atomic_concern_source as concern_module

    router = DirectTextRouter("private static int credits = 0;")
    generator = SimpleNamespace(router=router)
    target = tmp_path / "Probe.java"
    original = "public final class Probe {\n    private Probe() {}\n}\n"

    class DummyExecutor:
        def __init__(self, **kwargs):
            self.call_coder = kwargs["call_coder"]

        def run(self):
            region = self.call_coder(
                [{"role": "user", "content": "selected semantic Java region"}]
            )
            assert region == "private static int credits = 0;\n"
            return {
                "source": original,
                "summary": "semantic Java",
                "concern_count": 1,
                "repair_count": 0,
            }

    monkeypatch.setattr(concern_module, "AtomicConcernExecutor", DummyExecutor)

    module = SimpleNamespace(
        module_id="probe",
        kind="custom_java",
        required_gates=("target_compile",),
    )
    context = SimpleNamespace(
        root=tmp_path,
        target=target,
        relative="Probe.java",
        symbol="Probe",
        original=original,
        original_bytes=None,
        task={},
        section="algorithm",
        concerns=({"concern": "transitions"},),
        host_grounding={},
        dependency_context="",
        require_initialize=False,
        compiler=SimpleNamespace(compile_java=lambda _root: None),
        expected_package="",
        ir_contract={"public_api": []},
        module=module,
        target_existed=False,
        before_sha="sha256:before",
    )

    result = _run_atomic_ir_generation(generator, context)

    assert result["generation_verification"]["mode"] == (
        "gradle_compile_java_semantic_concerns"
    )
    assert len(router.calls) == 1
    assert router.calls[0][2]["enable_tools"] is False
    assert router.calls[0][2]["force_non_thinking"] is True
