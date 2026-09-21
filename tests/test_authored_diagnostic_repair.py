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
    ModelConfigurationError,
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
    legacy_repair_arguments = []

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
                assert "current_source" not in payload
                assert (
                    payload["current_source_sha256"]
                    == hashlib.sha256(source.encode("utf-8")).hexdigest()
                )
                window = payload["repair_window"]
                assert window["old"] == "MISSING"
                assert window["old"] != source
                assert any("MISSING" in d["message"] for d in payload["diagnostics"])
                assert all(d["path"] == target for d in payload["diagnostics"])
                parameters = request.tools[0]["function"]["parameters"]
                assert parameters["required"] == ["new"]
                assert parameters["additionalProperties"] is False
                assert set(parameters["properties"]) == {"new"}
                assert "host-owned" in guidance
                corrected = window["old"].replace("MISSING", "1")
                repairs.append(target)
                # Emulate a legacy/non-validating adapter that still returns the stale
                # model-owned fields seen in the production failure. The loop must strip
                # them and bind the live target/operation itself before runtime execution.
                arguments = {
                    "operation": "replace_exact",
                    "path": target,
                    "old": source + " ",
                    "new": corrected,
                }
                legacy_repair_arguments.append(dict(arguments))
                name = "apply_source_edit"
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
                    assert set(arguments) == {"operation", "path", "old", "new", "count"}
                    assert arguments["operation"] == "replace_exact"
                    assert arguments["count"] == 1
                    current = path.read_bytes().decode()
                    assert arguments["old"] in current
                    text = current.replace(arguments["old"], arguments["new"], 1)
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
                assert arguments.get("relative_files") == [
                    "src/main/java/demo/Next.java"
                ], (
                    "authored bounded-root generation must verify only the current "
                    "materialized Java target; sibling fragment errors belong to the "
                    "outer project build gate"
                )
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
    assert repairs == []
    assert calls == [
        "search_code_rag",
        "apply_source_edit",
        "java_diagnostics",
    ]
    assert all("MISSING" in (tmp_path / target).read_text() for target in targets)
    assert legacy_repair_arguments
    assert all("old" in item for item in legacy_repair_arguments)


def test_atomic_output_recovery_keeps_host_bound_repair_shape() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "parameters": {
                "type": "object",
                "properties": {"new": {"type": "string"}},
                "required": ["new"],
                "additionalProperties": False,
            },
        },
    }
    instruction = loop._atomic_output_recovery_instruction(
        GenerationRequest(tools=(schema,))
    )
    assert "replacement text" in instruction
    assert "bounded verifier repair window" in instruction
    assert "new argument" in instruction
    assert "operation, path, old text" in instruction
    assert "complete source file" in instruction


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



def test_authored_repair_fixed_point_stops_non_improving_rewrite_loop(
    tmp_path: Path,
) -> None:
    """Equal-error rewrites are rolled back and converge to a fixed-point failure."""

    target = "src/main/java/demo/First.java"
    source_a = "package demo; public class First { int value = MISSING_A; }\n"
    source_b = source_a.replace("MISSING_A", "MISSING_B")
    source_c = source_a.replace("MISSING_A", "MISSING_C")
    target_path = tmp_path / target
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_a, encoding="utf-8", newline="")

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
    runtime_calls: list[tuple[str, dict]] = []
    model_repairs = 0

    class Adapter:
        def generate_turn(self, request):
            nonlocal model_repairs
            if not runtime_calls:
                name = "search_code_rag"
                arguments = {"query": "First current workspace"}
            elif [name for name, _ in runtime_calls] == ["search_code_rag"]:
                name = "apply_source_edit"
                arguments = {
                    "operation": "replace_exact",
                    "path": target,
                    "old": "MISSING_A",
                    "new": "MISSING_B",
                }
            else:
                assert {
                    schema["function"]["name"] for schema in request.tools
                } == {"apply_source_edit"}
                model_repairs += 1
                name = "apply_source_edit"
                arguments = {"new": "MISSING_C"}
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{len(runtime_calls)}-{model_repairs}",
                        name=name,
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments),
                    ),
                )
            )

    class Runtime:
        workspace_root = tmp_path

        def call(self, stage, name, arguments):
            del stage
            runtime_calls.append((name, dict(arguments)))
            if name == "search_code_rag":
                return {
                    "hits": [{"path": target, "text": target_path.read_text()}],
                    "receipt": {"status": "FOUND", "result_count": 1},
                }
            if name == "apply_source_edit":
                before = target_path.read_bytes()
                current = before.decode("utf-8")
                if "old" in arguments:
                    updated = current.replace(arguments["old"], arguments["new"], 1)
                else:
                    updated = arguments["new"]
                target_path.write_text(updated, encoding="utf-8", newline="")
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "path": target,
                            "before_sha256": hashlib.sha256(before).hexdigest(),
                            "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
                        }
                    ],
                }
            if name == "java_diagnostics":
                current = target_path.read_text()
                unresolved = next(
                    (
                        marker
                        for marker in ("MISSING_A", "MISSING_B", "MISSING_C")
                        if marker in current
                    ),
                    None,
                )
                diagnostics = {}
                if unresolved is not None:
                    diagnostics[target_path.as_uri()] = [
                        {
                            "severity": 1,
                            "line": 1,
                            "code": "UNRESOLVED_VARIABLE",
                            "message": f"{unresolved} cannot be resolved to a variable",
                        }
                    ]
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
            raise AssertionError(name)

    payload = {
        "phase": "implement_authored_design",
        "task": "Repair the existing authored Java source",
        "module": module.config,
        "initial_exact_source_context": {
            "mode": "retrieve_current_authored_fragment_with_tools"
        },
        "authored_execution": {
            "schema_version": "mmm/authored-plan-fragment-v1",
            "fragment_index": 1,
            "fragment_count": 1,
            "source_text_sha256": "sha256:" + "b" * 64,
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
        for name in ("apply_source_edit", "search_code_rag", "java_diagnostics")
    )

    token = CURRENT_MUTATION_AUTHORITY.set(authority.mutation_authority)
    envelope_token = _CURRENT_AUTHORITY.set(authority)
    try:
        with pytest.raises(
            ModelConfigurationError,
            match="VERIFICATION_REPAIR_FIXED_POINT",
        ):
            loop.generate_with_tools(
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

    assert model_repairs == 2
    assert target_path.read_text() == source_b
    verifier_calls = [name for name, _ in runtime_calls if name == "java_diagnostics"]
    assert len(verifier_calls) == 3
    rollback_calls = [
        arguments
        for name, arguments in runtime_calls
        if (
            name == "apply_source_edit"
            and arguments.get("operation") == "replace_exact"
            and arguments.get("old") == source_c
            and arguments.get("new") == source_b
            and arguments.get("count") == 1
        )
    ]
    assert len(rollback_calls) == 2



def test_diagnostic_snapshot_preferred_path_never_switches_to_sibling(tmp_path: Path) -> None:
    from minecraft_mod_ai.generation_diagnostic_repair import (
        read_authorized_diagnostic_source,
    )
    from minecraft_mod_ai.mutation_authority import MutationAuthority

    first = tmp_path / "src/main/java/demo/First.java"
    second = tmp_path / "src/main/java/demo/Second.java"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("class First { Missing a; }\n", encoding="utf-8")
    second.write_text("class Second { Missing b; }\n", encoding="utf-8")
    snapshot = read_authorized_diagnostic_source(
        [
            {"uri": second.as_uri(), "severity": 1, "message": "Second broken"},
            {"uri": first.as_uri(), "severity": 1, "message": "First broken"},
        ],
        tmp_path,
        MutationAuthority.bounded_roots(),
        preferred_path="src/main/java/demo/First.java",
    )
    assert snapshot is not None
    assert snapshot["path"] == "src/main/java/demo/First.java"
    assert all(
        item["path"] == "src/main/java/demo/First.java"
        for item in snapshot["diagnostics"]
    )
