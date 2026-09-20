from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as loop
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
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


@pytest.mark.parametrize("unrepairable_errors_first", [False, True])
def test_authored_diagnostics_repair_local_files_without_external_discovery(
    tmp_path: Path,
    unrepairable_errors_first: bool,
) -> None:
    """A prior RAG call must not strand known source defects in external discovery."""
    targets = ["src/main/java/demo/First.java", "src/main/java/demo/Second.java"]
    sources = [
        "package demo; public class First { int value = MISSING; }\n",
        "package demo; public class Second { int value = MISSING; }\n",
    ]
    for target, source in zip(targets, sources, strict=True):
        path = tmp_path / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8", newline="")
    module = SimpleNamespace(
        module_id="space",
        kind="custom_java",
        config={
            "authored_plan": {
                "schema_version": "mmm/authored-plan-v1",
                "requested_prompt": "space mode",
                "text": "space mechanics",
            },
        },
    )
    authority = compile_direct_task_mutation_authority(module)
    assert authority is not None
    calls = []
    repairs = []

    class Adapter:
        def generate_turn(self, request):
            names = {s["function"]["name"] for s in request.tools}
            if not calls:
                name, arguments = "search_code_rag", {"query": "current workspace"}
            elif calls == ["search_code_rag"]:
                name, arguments = (
                    "apply_source_edit",
                    {
                        "operation": "create_file",
                        "path": "src/main/java/demo/Next.java",
                        "content": "package demo; public class Next {}\n",
                    },
                )
            else:
                assert names == {"apply_source_edit"}, (
                    "known local defects need a bound repair, not external discovery"
                )
                repair_messages = [
                    m["content"]
                    for m in request.messages
                    if str(m.get("content", "")).startswith("MMM_CORE_VERIFIER_REPAIR_")
                ]
                assert len(repair_messages) == 1, (
                    "old full-source repair prompts must be retired"
                )
                guidance = repair_messages[0]
                payload = json.loads(guidance.rsplit("\n", 1)[-1])
                target = targets[len(repairs)]
                source = (tmp_path / target).read_bytes().decode("utf-8")
                assert payload["target_path"] == target
                assert payload["current_source"] == source
                assert (
                    payload["current_source_sha256"]
                    == hashlib.sha256(source.encode()).hexdigest()
                )
                assert any("MISSING" in d["message"] for d in payload["diagnostics"])
                assert all(d["path"] == target for d in payload["diagnostics"])
                assert loop._existing_repair_context(request.messages) == payload
                properties = request.tools[0]["function"]["parameters"]["properties"]
                assert properties["path"]["enum"] == [target]
                assert "replace_exact" in properties["operation"]["enum"]
                assert "create_file" not in properties["operation"]["enum"]
                assert "create_file/create" not in guidance
                for message in request.messages:
                    if message["role"] == "system":
                        assert "same-path create_file/create" not in message["content"]
                        assert "Future repairs stay on this exact path" not in message["content"]
                repairs.append(target)
                name, arguments = (
                    "apply_source_edit",
                    {
                        "operation": "replace_exact",
                        "path": target,
                        "old": "MISSING",
                        "new": "1",
                    },
                )
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{len(calls)}",
                        name=name,
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments),
                    ),
                )
            )

    class Runtime:
        workspace_root = tmp_path

        def call(self, stage, name, arguments):
            calls.append(name)
            if name == "search_code_rag":
                return {
                    "hits": [{"path": targets[0], "text": sources[0]}],
                    "receipt": {"status": "FOUND", "result_count": 1},
                }
            if name == "apply_source_edit":
                path = tmp_path / arguments["path"]
                if arguments["operation"] == "create_file":
                    text = arguments["content"]
                else:
                    text = (
                        path.read_bytes()
                        .decode()
                        .replace(arguments["old"], arguments["new"])
                    )
                path.write_text(text, encoding="utf-8", newline="")
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "path": arguments["path"],
                            "after_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    ],
                }
            if name == "java_diagnostics":
                assert not arguments.get("relative_files"), (
                    "bounded-root repair must recheck all project errors"
                )
                diagnostics = {
                    (tmp_path / target).as_uri(): [
                        {
                            "severity": 1,
                            "line": 1,
                            "code": "UNRESOLVED_VARIABLE",
                            "message": "MISSING cannot be resolved to a variable",
                        }
                    ]
                    for target in targets
                    if "MISSING" in (tmp_path / target).read_text()
                }
                if diagnostics and unrepairable_errors_first:
                    diagnostics = {
                        (tmp_path / "build/generated/Unavailable.java").as_uri(): [
                            {"severity": 1, "message": "CASCADED_ERROR " + "x" * 800}
                            for _ in range(12)
                        ],
                        **diagnostics,
                    }
                return {
                    "schema_version": "mmm/java-diagnostics-v3",
                    "complete": True,
                    "skipped": False,
                    "session_id": "session",
                    "model_id": "model",
                    "verification_scope": "full",
                    "error_count": len(diagnostics),
                    "warning_count": 0,
                    "diagnostics": diagnostics,
                }
            raise AssertionError(f"unrelated recovery tool called: {name}")

    payload = {
        "phase": "implement_authored_design",
        "task": "Implement the next saved fragment",
        "module": module.config,
        "initial_exact_source_context": {
            "mode": "retrieve_current_authored_fragment_with_tools"
        },
        "authored_execution": {
            "schema_version": "mmm/authored-plan-fragment-v1",
            "fragment_index": 2,
            "fragment_count": 3,
            "source_text_sha256": "sha256:" + "a" * 64,
        },
    }
    tools = tuple(
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": (
                    SOURCE_EDIT_SCHEMA
                    if name == "apply_source_edit"
                    else {"type": "object", "properties": {}}
                ),
            },
        }
        for name in (
            "apply_source_edit",
            "search_code_rag",
            "java_diagnostics",
            "inspect_modrinth_project",
            "read_reuse_source",
        )
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
            request=GenerationRequest(
                messages=({"role": "user", "content": json.dumps(payload)},),
                tools=tools,
            ),
            runtime=Runtime(),
            stage="generation",
            role="coder",
        )
    finally:
        _CURRENT_AUTHORITY.reset(envelope_token)
        CURRENT_MUTATION_AUTHORITY.reset(token)
    assert "passed" in json.loads(result)["summary"]
    assert repairs == targets
    assert calls == [
        "search_code_rag",
        "apply_source_edit",
        "java_diagnostics",
        "apply_source_edit",
        "java_diagnostics",
        "apply_source_edit",
        "java_diagnostics",
    ]
    assert all("MISSING" not in (tmp_path / target).read_text() for target in targets)


@pytest.mark.parametrize(
    "kind", ["outside", "build", "missing", "remote_uri", "no_authority", "oversized"]
)
def test_diagnostic_source_does_not_expand_authority(tmp_path: Path, kind: str) -> None:
    from minecraft_mod_ai.generation_diagnostic_repair import (
        read_authorized_diagnostic_source,
    )
    from minecraft_mod_ai.mutation_authority import MutationAuthority

    root = tmp_path / "project"
    root.mkdir()
    target = root / "src/main/java/demo/Broken.java"
    if kind == "outside":
        target = tmp_path / "Outside.java"
    elif kind == "build":
        target = root / "build.gradle"
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind != "missing":
        target.write_text(
            "class Broken {}" + ("x" * 100_000 if kind == "oversized" else "")
        )
    uri = (
        "https://example.invalid/Broken.java"
        if kind == "remote_uri"
        else target.as_uri()
    )
    assert (
        read_authorized_diagnostic_source(
            [{"uri": uri, "severity": 1, "message": "broken"}],
            root,
            None if kind == "no_authority" else MutationAuthority.bounded_roots(),
        )
        is None
    )


def test_diagnostic_snapshot_preserves_crlf_hash_and_uri_path(tmp_path: Path) -> None:
    from minecraft_mod_ai.generation_diagnostic_repair import (
        read_authorized_diagnostic_source,
    )
    from minecraft_mod_ai.mutation_authority import MutationAuthority

    target = tmp_path / "src/main/java/demo/Has Space.java"
    target.parent.mkdir(parents=True)
    raw = b"package demo;\r\nclass Broken { int value = MISSING; }\r\n"
    target.write_bytes(raw)
    snapshot = read_authorized_diagnostic_source(
        [{"uri": target.as_uri(), "severity": 1, "message": "MISSING"}],
        tmp_path,
        MutationAuthority.bounded_roots(),
    )
    assert snapshot is not None
    assert snapshot["path"] == "src/main/java/demo/Has Space.java"
    assert snapshot["source"] == raw.decode()
    assert snapshot["sha256"] == hashlib.sha256(raw).hexdigest()
