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


def test_authored_recovery_fits_after_failed_and_empty_evidence_routes() -> None:
    """Exercise the loop through VERIFY->RECOVER and repeated recovery turns."""
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
    module = SimpleNamespace(
        module_id="space-mode",
        kind="custom_java",
        config={
            "authored_plan": {
                "schema_version": "mmm/authored-plan-v1",
                "requested_prompt": "space mechanics",
                "text": "approved behavior " * 650,
            },
        },
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
            "inspect_modrinth_project",
            "external_mcp_capabilities",
            "external_mcp_schema",
            "external_mcp_call",
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
            task = next(
                json.loads(m["content"])
                for m in request.messages
                if m["role"] == "user"
            )
            assert task["task"] == original_task["task"]
            assert task["module"] == original_task["module"]
            assert any(
                m.get("content") == "Keep user approval and write boundaries."
                for m in request.messages
            )
            if step in (3, 4, 5):
                rendered = json.dumps(request.messages)
                assert diagnostic in rendered
                assert "mmm/verifier-recovery-handoff-v1" in rendered
                assert request.tool_choice == "required"
                assert "apply_source_edit" not in {
                    s["function"]["name"] for s in request.tools
                }
            name, arguments = {
                1: ("search_code_rag", {"query": "current workspace"}),
                2: (
                    "apply_source_edit",
                    {"operation": "create_file", "path": target, "content": source},
                ),
                3: ("read_reuse_source", {"path": "src/main/java/"}),
                4: ("search_project_rag", {"query": "demo.MissingApi"}),
                5: ("java_workspace_symbols", {"query": "SpaceModeMod"}),
                6: (
                    "apply_source_edit",
                    {
                        "operation": "replace_exact",
                        "path": target,
                        "old": source,
                        "new": repaired_source,
                    },
                ),
            }[step]
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
                errors = [
                    {
                        "path": target,
                        "line": 2,
                        "severity": 1,
                        "code": "IMPORT_NOT_FOUND",
                        "message": diagnostic,
                    }
                ]
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
            if name == "read_reuse_source":
                raise RuntimeError("Reuse source path escaped the approved donor root.")
            if name == "search_project_rag":
                return {"hits": [], "receipt": {"status": "EMPTY", "result_count": 0}}
            if name == "java_workspace_symbols":
                return {
                    "symbols": [
                        {"name": "SpaceModeMod", "path": target, "source": source}
                    ]
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
    assert len(observed) == 6
    assert runtime_calls == [
        "search_code_rag",
        "apply_source_edit",
        "java_diagnostics",
        # read_reuse_source is no longer in the pinned-source recovery frontier;
        # the model's attempted call is rejected before runtime execution.
        "search_project_rag",
        "java_workspace_symbols",
        "apply_source_edit",
        "java_diagnostics",
    ]
    assert request.messages[1]["content"] == routing



def test_authored_stale_precondition_rebinds_live_workspace_target(tmp_path) -> None:
    target = "src/main/java/xyz/mods/spacemode/core/SyncManager.java"
    live_source = (
        "package xyz.mods.spacemode.core;\n"
        "public class SyncManager { int value = 1; }\n"
    )
    target_file = tmp_path / target
    target_file.parent.mkdir(parents=True)
    target_file.write_text(live_source, encoding="utf-8")

    module = SimpleNamespace(
        module_id="space-mode",
        kind="custom_java",
        config={"authored_plan": {"schema_version": "mmm/authored-plan-v1"}},
    )
    authority = compile_direct_task_mutation_authority(module)
    assert authority is not None
    payload = {
        "phase": "implement_authored_design",
        "task": "Update the current sync fragment.",
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
    tools = tuple(
        _schema(name)
        for name in (
            "search_code_rag",
            "apply_source_edit",
            "java_diagnostics",
            "search_project_rag",
        )
    )
    observed = []
    runtime_calls = []

    class Adapter:
        def generate_turn(self, request):
            observed.append(request)
            step = len(observed)
            if step == 1:
                name = "search_code_rag"
                arguments = {"query": "SyncManager.java current workspace source"}
            elif step == 2:
                name = "apply_source_edit"
                arguments = {
                    "operation": "replace_exact",
                    "path": target,
                    "old": "stale source body",
                    "new": "still stale",
                }
            else:
                rendered = json.dumps(request.messages)
                assert "MMM_EXISTING_TARGET_REFRESH_V1" in rendered
                assert "int value = 1;" in rendered
                name = "apply_source_edit"
                arguments = {
                    "operation": "replace_exact",
                    "path": target,
                    "old": "int value = 1;",
                    "new": "int value = 2;",
                }
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
        workspace_root = tmp_path

        def call(self, stage, name, arguments):
            assert stage == "generation"
            runtime_calls.append(name)
            if name == "search_code_rag":
                return {
                    "hits": [
                        {
                            "path": target,
                            "source_path": target,
                            "text": "stale source body",
                        }
                    ],
                    "receipt": {"status": "FOUND", "result_count": 1},
                }
            if name == "apply_source_edit":
                if arguments.get("old") == "stale source body":
                    raise RuntimeError(
                        f"Exact source-edit precondition failed for {target}: "
                        "expected 1 matches, found 0"
                    )
                current = target_file.read_text(encoding="utf-8")
                old = arguments["old"]
                assert current.count(old) == 1
                updated = current.replace(old, arguments["new"], 1)
                target_file.write_text(updated, encoding="utf-8")
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [{"path": target, "after_sha256": "sha256:new"}],
                }
            if name == "java_diagnostics":
                assert arguments.get("relative_files") == [target]
                return {
                    "schema_version": "mmm/java-diagnostics-v3",
                    "complete": True,
                    "skipped": False,
                    "session_id": "session",
                    "model_id": "model",
                    "verification_scope": "target",
                    "error_count": 0,
                    "warning_count": 0,
                    "diagnostics": {},
                }
            raise AssertionError(name)

    request = GenerationRequest(
        messages=({"role": "user", "content": json.dumps(payload)},),
        tools=tools,
    )
    token = CURRENT_MUTATION_AUTHORITY.set(authority.mutation_authority)
    envelope_token = _CURRENT_AUTHORITY.set(authority)
    try:
        result = loop.generate_with_tools(
            SimpleNamespace(_agent_require_fresh_evidence=False),
            config=SimpleNamespace(
                adapter="test",
                max_context=32768,
                max_input_tokens=0,
                max_new_tokens=512,
            ),
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
    assert "int value = 2;" in target_file.read_text(encoding="utf-8")
    assert runtime_calls == [
        "search_code_rag",
        "apply_source_edit",
        "apply_source_edit",
        "java_diagnostics",
    ]
    assert "search_project_rag" not in runtime_calls
