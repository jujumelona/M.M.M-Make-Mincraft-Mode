from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import progress_aware_tool_loop as loop
from minecraft_mod_ai import small_model_task_capsule_contract as capsules
from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse, ToolCall


def test_target_compile_still_requires_fresh_java_evidence_before_mutation(
    monkeypatch,
) -> None:
    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    monkeypatch.setattr(
        capsules,
        "current_task_required_gates",
        lambda: ("target_compile",),
    )

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate_turn(self, request):
            self.calls += 1
            names = {item["function"]["name"] for item in request.tools}
            if self.calls == 1:
                assert names == {"search_code_rag"}
                arguments = {"query": "Fabric item registration exact API"}
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="read-1",
                            name="search_code_rag",
                            arguments=arguments,
                            raw_arguments=json.dumps(arguments),
                        ),
                    )
                )
            assert self.calls == 2
            assert names == {"apply_source_edit"}
            arguments = {
                "operation": "create_file",
                "path": target,
                "content": (
                    "package dev.mmm.debugfixture; "
                    "public final class DebugToken {}\n"
                ),
            }
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="edit-1",
                        name="apply_source_edit",
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments),
                    ),
                )
            )

    runtime_calls: list[str] = []

    class Runtime:
        def call(self, stage, name, _arguments):
            assert stage == "generation"
            runtime_calls.append(name)
            if name == "search_code_rag":
                return {
                    "schema_version": "mmm/code-rag-result-v1",
                    "hits": [
                        {
                            "path": "official/FabricExample.java",
                            "text": (
                                "import net.minecraft.registry.Registry; "
                                "import net.minecraft.item.Item;"
                            ),
                        }
                    ],
                }
            if name == "apply_source_edit":
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "operation": "create",
                            "path": target,
                            "before_sha256": None,
                            "after_sha256": "sha256:" + "1" * 64,
                        }
                    ],
                }
            if name == "target_compile":
                return {
                    "schema_version": "mmm/generation-target-compile-v1",
                    "status": "PASS",
                    "target_path": target,
                    "diagnostics": [],
                }
            raise AssertionError(name)

    request = GenerationRequest(
        messages=(
            {
                "role": "developer",
                "content": json.dumps(
                    {
                        "primary_path": target,
                        "writable_paths": [target],
                        "reuse_action": "fresh",
                    }
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": "implement_module",
                        "task": "Implement the approved debug token.",
                    }
                ),
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "search_code_rag",
                    "description": "retrieve exact Java/API evidence",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "description": "edit source",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )

    adapter = Adapter()
    result = loop.generate_with_tools(
        SimpleNamespace(_agent_require_fresh_evidence=True),
        config=SimpleNamespace(
            adapter="test",
            max_context=32768,
            max_input_tokens=0,
            max_new_tokens=512,
        ),
        adapter=adapter,
        request=request,
        runtime=Runtime(),
        stage="generation",
        role="coder",
    )

    payload = json.loads(result)
    assert "passed generation-time host verification" in payload["summary"]
    assert adapter.calls == 2
    assert runtime_calls == ["search_code_rag", "apply_source_edit", "target_compile"]
