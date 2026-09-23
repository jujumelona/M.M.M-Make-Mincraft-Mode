from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as loop
from minecraft_mod_ai import small_model_task_capsule_contract as capsules
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ToolCall,
)
from minecraft_mod_ai.mutation_authority import (
    CURRENT_MUTATION_AUTHORITY,
    MutationAuthority,
)


@pytest.mark.parametrize("status", ["FAIL", "PENDING"])
def test_host_compile_diagnostics_reach_recovery_without_a_model_verify_turn(status):
    state = loop.HostRunState(
        phase=loop.LoopPhase.RECOVER,
        validation_status=status,
        latest_verifier_tool="target_compile",
        latest_verifier_errors=(
            {"message": "package net.minecraft.item does not exist"},
        ),
        latest_verifier_fingerprint="compiler-failure",
        repair_evidence_route="official_api",
    )
    messages = [{"role": "user", "content": "Implement approved feature"}]
    loop._sync_phase_tool_transcript(
        messages, state=state, last_prompt_phase=loop.LoopPhase.ACT, stage="generation"
    )
    handoff = messages[-1]["content"]
    assert "package net.minecraft.item does not exist" in handoff
    assert "target_compile" in handoff


def test_compile_api_recovery_requires_evidence_even_when_initial_search_was_skipped():
    state = loop.HostRunState(
        phase=loop.LoopPhase.RECOVER,
        validation_status="PENDING",
        repair_evidence_route="official_api",
        semantic_fresh_java=True,
        latest_verifier_errors=(
            {"message": "package net.minecraft.item does not exist"},
        ),
    )
    state.record_evidence(
        {"schema_version": "mmm/rag-result-v2", "sources": [{"title": "Items"}]},
        usable=True,
    )
    assert not loop._target_evidence_ready(
        state, require_rag=False, fresh_java_target=True, compile_backed_java=True
    )


@pytest.mark.parametrize("compiler", ["receipt", "javac"])
def test_compile_recovery_forces_live_retriever_then_recompiles_authored_source(
    monkeypatch, tmp_path, compiler
):
    javac = shutil.which("javac") if compiler == "javac" else None
    if compiler == "javac" and javac is None:
        pytest.skip("Java compiler is not installed")
    target = "src/main/java/demo/Feature.java"
    source = tmp_path / target
    source.parent.mkdir(parents=True)
    baseline = (
        "package demo;\npublic class Feature { public static void initialize() {} }\n"
    )
    broken = baseline.replace(
        "public class", "import net.minecraft.item.Item;\npublic class"
    )
    fixed = baseline.replace(
        "initialize() {}", "initialize() { System.out.println(1); }"
    )
    source.write_text(baseline, encoding="utf-8", newline="")
    compiler_outcomes = []
    monkeypatch.setattr(
        capsules, "current_task_required_gates", lambda: ("target_compile",)
    )
    monkeypatch.setattr(capsules, "current_task_reuse_action", lambda: "fresh")
    observed = []

    class Adapter:
        def generate_turn(self, request):
            name = request.tools[0]["function"]["name"]
            observed.append(name)
            if len(observed) == 1:
                args = {"new": broken}
            elif name.startswith("search_"):
                assert request.tool_choice == {
                    "type": "function",
                    "function": {"name": name},
                }
                assert request.parallel_tool_calls is False
                assert "package net.minecraft.item does not exist" in str(
                    request.messages
                )
                args = {"query": "net.minecraft.item.Item correct API package"}
            else:
                assert observed == [
                    "apply_source_edit",
                    "search_project_rag",
                    "search_code_rag",
                    "apply_source_edit",
                ]
                args = {"new": fixed}
            return GenerationResponse(
                tool_calls=(ToolCall(id=str(len(observed)), name=name, arguments=args),)
            )

    calls = []

    class Runtime:
        workspace_root = tmp_path

        def call(self, stage, name, args):
            calls.append(name)
            if name == "apply_source_edit":
                current = source.read_bytes().decode("utf-8")
                assert args["path"] == target and current.count(args["old"]) == 1
                updated = current.replace(args["old"], args["new"], 1)
                source.write_text(updated, encoding="utf-8", newline="")
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "operation": "replace",
                            "path": target,
                            "before_sha256": hashlib.sha256(
                                current.encode()
                            ).hexdigest(),
                            "after_sha256": hashlib.sha256(
                                updated.encode()
                            ).hexdigest(),
                        }
                    ],
                }
            if name == "target_compile":
                bad = "net.minecraft.item" in source.read_text(encoding="utf-8")
                diagnostic = "package net.minecraft.item does not exist"
                if javac is not None:
                    compiled = subprocess.run(
                        [javac, "-J-Duser.language=en", "-d", str(tmp_path / "classes"), str(source)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    bad = compiled.returncode != 0
                    diagnostic = compiled.stderr
                compiler_outcomes.append(bad)
                return {
                    "schema_version": "mmm/generation-target-compile-v1",
                    "status": "FAIL" if bad else "PASS",
                    "target_path": target,
                    "diagnostics": [
                        {
                            "path": target,
                            "severity": 1,
                            "source": "javac",
                            "line": 2,
                            "message": diagnostic,
                        }
                    ]
                    if bad
                    else [],
                }
            if name == "search_project_rag":
                return {
                    "schema_version": "mmm/rag-result-v2",
                    "sources": [
                        {"title": "Item guide", "url": "https://example.test/guide"}
                    ],
                }
            if name == "search_code_rag":
                return {
                    "schema_version": "mmm/code-rag-result-v1",
                    "hits": [
                        {
                            "path": "net/minecraft/world/item/Item.java",
                            "text": "package net.minecraft.world.item; public class Item {}",
                            "sha256": "api-source",
                        }
                    ],
                }
            raise AssertionError(name)

    tools = tuple(
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}}
                    if name.startswith("search_")
                    else {},
                },
            },
        }
        for name in ("apply_source_edit", "search_project_rag", "search_code_rag")
    )
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
                        "task": "Implement the approved feature.",
                    }
                ),
            },
        ),
        tools=tools,
    )
    token = CURRENT_MUTATION_AUTHORITY.set(MutationAuthority.exact((target,)))
    try:
        result = loop.generate_with_tools(
            SimpleNamespace(_agent_require_fresh_evidence=True),
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
        CURRENT_MUTATION_AUTHORITY.reset(token)
    assert "passed generation-time host verification" in json.loads(result)["summary"]
    assert source.read_text(encoding="utf-8") == fixed
    assert calls.count("target_compile") == 2
    assert compiler_outcomes == [True, False]
    if javac is not None:
        assert (tmp_path / "classes/demo/Feature.class").is_file()
