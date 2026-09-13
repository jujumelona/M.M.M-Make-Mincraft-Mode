from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall
from minecraft_mod_ai.model_router import ModelRouter


class _Registry:
    def __init__(self) -> None:
        self.config = SimpleNamespace(adapter="llama_cpp", exclusive_gpu=False)

    def load_profile(self, profile: str) -> None:
        assert profile == "test"

    def role(self, profile: str, role: str):
        assert profile == "test"
        assert role == "coder"
        return self.config


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string"},
                    "content": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
    }


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def tool_schemas(self, stage: str):
        assert stage == "generation"
        return (_schema("apply_source_edit"), _schema("java_diagnostics"))

    def call(self, stage: str, name: str, arguments):
        payload = dict(arguments)
        self.calls.append((stage, name, payload))
        if name == "apply_source_edit":
            return {
                "schema_version": "mmm/source-patch-receipt-v1",
                "status": "APPLIED",
                "operations": [
                    {
                        "before_sha256": "sha256:before",
                        "after_sha256": "sha256:after",
                    }
                ],
            }
        if name == "java_diagnostics":
            return {"status": "PASS", "diagnostics": []}
        raise AssertionError(f"unexpected tool call: {name}")


class _Adapter:
    def __init__(self) -> None:
        self.requests = []

    def generate_turn(self, request):
        self.requests.append(request)
        index = len(self.requests)
        names = [item["function"]["name"] for item in request.tools]
        if index == 1:
            assert names == ["apply_source_edit"]
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="mutate_1",
                        name="apply_source_edit",
                        arguments={
                            "path": "src/main/java/Example.java",
                            "operation": "create_file",
                            "content": "public final class Example {}",
                        },
                        raw_arguments=json.dumps(
                            {
                                "path": "src/main/java/Example.java",
                                "operation": "create_file",
                                "content": "public final class Example {}",
                            }
                        ),
                    ),
                )
            )
        if index == 2:
            # Generation verification is now host-owned. The coder does not spend a
            # redundant inference turn merely selecting java_diagnostics; after the
            # host verifier passes, the next model turn is the terminal no-tool turn.
            assert request.tools == ()
            assert request.tool_choice is None
            return GenerationResponse(content="verified implementation complete")
        raise AssertionError("verifier was invoked again after a trusted PASS")


def test_verifier_pass_finalizes_without_repeating_verifier(monkeypatch) -> None:
    adapter = _Adapter()
    runtime = _Runtime()
    monkeypatch.setattr(
        ModelRouter,
        "_new_text_adapter",
        staticmethod(lambda config, *, role: adapter),
    )
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )
    request = {
        "task": "implement_module",
        "operation": "create_file",
        "path": "src/main/java/Example.java",
        "request": "implement_module",
    }

    result = router.generate_text(
        "coder",
        [{"role": "user", "content": json.dumps(request)}],
    )

    assert result == "verified implementation complete"
    assert [name for _, name, _ in runtime.calls] == [
        "apply_source_edit",
        "java_diagnostics",
    ]
    assert len(adapter.requests) == 2
    final_instruction = str(adapter.requests[-1].messages[-1]["content"])
    assert "verification passed" in final_instruction.casefold()
