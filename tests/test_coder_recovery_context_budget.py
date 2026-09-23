from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import model_context_budget
from minecraft_mod_ai import progress_aware_tool_loop as loop
from minecraft_mod_ai.agent_capability_context import build_agent_capability_context
from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    _CURRENT_AUTHORITY,
    compile_direct_task_mutation_authority,
)
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ToolCall,
)
from minecraft_mod_ai.mutation_authority import CURRENT_MUTATION_AUTHORITY


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }



def _exact_authored_module(module_id: str, target: str, *, text: str):
    symbol = target.rsplit("/", 1)[-1].removesuffix(".java")
    anchor = {
        "kind": "symbol",
        "locator": f"{target}#{symbol}",
        "status": "existing",
        "ownership": "host_test_exact",
        "module_id": module_id,
        "source_set": "main",
    }
    return SimpleNamespace(
        module_id=module_id,
        kind="custom_java",
        config={
            "authored_plan": {
                "schema_version": "mmm/authored-plan-v1",
                "requested_prompt": "space mechanics",
                "text": text,
            },
            "evidence_task": {
                "task_id": module_id,
                "semantic_outcome": "exercise exact authored recovery mechanics",
                "implementation_obligations": ["repair only the exact existing target"],
                "engineering_worksheet": {"objective": "exact authored recovery target"},
                "owned_anchors": [anchor],
                "production_bindings": [{
                    "task_ref": module_id,
                    "reuse_action": "fresh",
                    "owned_anchors": [anchor],
                }],
                "required_gates": ["target_compile"],
            },
        },
        required_gates=("target_compile",),
    )


def test_authored_recovery_fits_after_failed_and_empty_evidence_routes() -> None:
    """Exact authored recovery stays inside budget and uses the pinned repair tool."""

    target = "src/main/java/demo/SpaceModeMod.java"
    source = (
        "package demo;\n"
        "import demo.MissingApi;\n"
        "public class SpaceModeMod {}\n"
    )
    repaired_source = (
        "package demo;\n"
        "public class SpaceModeMod {}\n"
    )
    diagnostic = "The import demo.MissingApi cannot be resolved"
    module = _exact_authored_module(
        "space-mode",
        target,
        text="approved behavior " * 650,
    )
    authority = compile_direct_task_mutation_authority(module)
    assert authority is not None
    tools = tuple(
        _schema(name)
        for name in (
            "search_code_rag",
            "apply_source_edit",
            "java_diagnostics",
            "read_reuse_source",
            "search_project_rag",
            "java_workspace_symbols",
        )
    )
    routing = build_agent_capability_context("generation", tools, model_role="coder")
    original_task = {
        "phase": "implement_authored_design",
        "task": "Implement the approved fragment.",
        "module": module.config,
        "initial_exact_source_context": {
            "mode": "retrieve_current_authored_fragment_with_tools",
        },
        "authored_execution": {
            "schema_version": "mmm/authored-plan-fragment-v1",
            "fragment_index": 2,
            "fragment_count": 3,
            "source_text_sha256": "sha256:" + "a" * 64,
        },
    }
    config = SimpleNamespace(
        adapter="llama_cpp",
        max_context=32768,
        max_input_tokens=22400,
        max_new_tokens=8192,
        extra={"runtime_context_default": 32768},
    )
    observed = []
    runtime_calls = []
    verification_count = 0

    class Adapter:
        def generate_turn(self, request):
            observed.append(request)
            step = len(observed)
            assert model_context_budget._canonical_size(request.messages) <= (
                model_context_budget.request_message_budget(config, request.tools)
            )
            if step == 1:
                name, arguments = "search_code_rag", {"query": "current workspace"}
            elif step == 2:
                name, arguments = (
                    "apply_source_edit",
                    {"operation": "create_file", "path": target, "content": source},
                )
            else:
                names = {schema["function"]["name"] for schema in request.tools}
                assert names == {"apply_source_edit"}
                rendered = json.dumps(request.messages)
                assert "host-pinned" in rendered or "host-owned" in rendered
                name, arguments = (
                    "apply_source_edit",
                    {
                        "operation": "replace_exact",
                        "path": target,
                        "old": source,
                        "new": repaired_source,
                    },
                )
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{step}",
                        name=name,
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments),
                    ),
                )
            )

    class Runtime:
        def call(self, stage, name, arguments):
            nonlocal verification_count
            assert stage == "generation"
            runtime_calls.append(name)
            if name == "search_code_rag":
                return {
                    "hits": [{"path": target, "source_path": target, "text": source}],
                    "receipt": {"status": "FOUND", "result_count": 1},
                }
            if name == "apply_source_edit":
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {"path": target, "after_sha256": "sha256:" + "b" * 64}
                    ],
                }
            if name == "java_diagnostics":
                verification_count += 1
                errors = [{
                    "path": target,
                    "line": 2,
                    "severity": 1,
                    "code": "IMPORT_NOT_FOUND",
                    "message": diagnostic,
                }]
                return {
                    "schema_version": "mmm/java-diagnostics-v3",
                    "complete": True,
                    "session_id": "session",
                    "model_id": "model",
                    "skipped": False,
                    "error_count": int(verification_count == 1),
                    "warning_count": 0,
                    "diagnostics": {target: errors} if verification_count == 1 else {},
                }
            raise AssertionError(name)

    request = GenerationRequest(
        messages=(
            {"role": "system", "content": "Keep user approval and write boundaries."},
            {"role": "system", "content": routing},
            {"role": "user", "content": json.dumps(original_task)},
        ),
        tools=tools,
    )
    token = CURRENT_MUTATION_AUTHORITY.set(authority.mutation_authority)
    envelope_token = _CURRENT_AUTHORITY.set(authority)
    try:
        result = loop.generate_with_tools(
            SimpleNamespace(_agent_require_fresh_evidence=False),
            config=config,
            adapter=Adapter(),
            request=request,
            runtime=Runtime(),
            stage="generation",
            role="coder",
        )
    finally:
        _CURRENT_AUTHORITY.reset(envelope_token)
        CURRENT_MUTATION_AUTHORITY.reset(token)

    assert "passed" in json.loads(result)["summary"]
    assert len(observed) == 3
    assert runtime_calls == [
        "search_code_rag",
        "apply_source_edit",
        "java_diagnostics",
        "apply_source_edit",
        "java_diagnostics",
    ]
    assert request.messages[1]["content"] == routing
def test_authored_stale_precondition_cannot_rebind_outside_frozen_target(tmp_path) -> None:
    target = "src/main/java/xyz/mods/spacemode/core/SyncManager.java"
    target_file = tmp_path / target
    target_file.parent.mkdir(parents=True)
    target_file.write_text(
        "package xyz.mods.spacemode.core;\n"
        "public class SyncManager { int value = 1; }\n",
        encoding="utf-8",
    )

    module = _exact_authored_module(
        "space-mode",
        target,
        text="update current sync fragment",
    )
    authority = compile_direct_task_mutation_authority(module)
    assert authority is not None

    token = CURRENT_MUTATION_AUTHORITY.set(authority.mutation_authority)
    envelope_token = _CURRENT_AUTHORITY.set(authority)
    try:
        assert loop._mutation_target_error(
            "apply_source_edit",
            {
                "operation": "replace_exact",
                "path": target,
                "old": "stale source body",
                "new": "still stale",
            },
            None,
        ) is None
        error = loop._mutation_target_error(
            "apply_source_edit",
            {
                "operation": "replace_exact",
                "path": "src/main/java/xyz/mods/spacemode/core/Other.java",
                "old": "x",
                "new": "y",
            },
            None,
        )
        assert error is not None
        assert "MUTATION_TARGET_DRIFT" in error
    finally:
        _CURRENT_AUTHORITY.reset(envelope_token)
        CURRENT_MUTATION_AUTHORITY.reset(token)

    assert "int value = 1;" in target_file.read_text(encoding="utf-8")
