from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import custom_generation_research as research
from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.model_adapters import GenerationRequest


class _InnerRouter:
    def __init__(self) -> None:
        self.calls = []
        self._agent_require_fresh_evidence = False

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, tuple(dict(message) for message in messages), dict(kwargs)))
        return "done"


class _ToolPathEngine:
    def __init__(self) -> None:
        self.ingested = []

    def ingest_code_owned_request(self, messages):
        self.ingested.append(tuple(dict(message) for message in messages))

    def initial_bundle(self):
        raise AssertionError("tool-enabled coder must not build a static research bundle")

    def receipt(self):
        return {"schema_version": "test", "status": "PASS"}


def test_tool_enabled_coder_leaves_research_retrieval_to_canonical_tool_loop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    inner = _InnerRouter()
    router = research._ResearchEvidenceRouter(
        inner,
        owner=SimpleNamespace(policy=None),
        project_root=tmp_path,
        module=None,
        minecraft_version="26.1",
        loader="fabric",
        mappings="",
    )
    engine = _ToolPathEngine()
    router._context = engine
    monkeypatch.setattr(
        research,
        "_sanitized_messages",
        lambda messages, **_kwargs: [dict(message) for message in messages],
    )
    messages = (
        {"role": "system", "content": "coder"},
        {"role": "user", "content": '{"phase":"implement_module"}'},
    )

    assert router.generate_text("coder", messages, enable_tools=True) == "done"

    assert len(engine.ingested) == 1
    assert len(inner.calls) == 1
    forwarded = inner.calls[0][1]
    assert forwarded == messages
    assert not any(
        str(message.get("content") or "").startswith("Host research context follows.")
        for message in forwarded
    )


def test_forced_act_projects_away_finished_research_and_routing_context() -> None:
    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    state = SimpleNamespace(
        phase=tool_loop.LoopPhase.ACT,
        mutation_context=tool_loop.TargetMutationContext(
            target_path=target,
            target_symbol="DebugToken",
            is_new_file=True,
            evidence_source="evidence_fresh_owned_anchor",
            writable_paths=(target,),
            creatable_paths=(target,),
            target_pinned=True,
        ),
    )
    messages = (
        {"role": "system", "content": "coder"},
        {
            "role": "system",
            "content": "Host research context follows.\n" + "x" * 20000,
        },
        {
            "role": "system",
            "content": "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n" + "y" * 20000,
        },
        {"role": "system", "content": "Repository branch policy: main only"},
        {"role": "developer", "content": '{"task_id":"debug_token"}'},
        {"role": "user", "content": '{"phase":"implement_module"}'},
    )

    projected = tool_loop._forced_act_messages(
        messages,
        state=state,
        require_rag=False,
        phase_names={"apply_source_edit"},
    )

    contents = [str(message.get("content") or "") for message in projected]
    assert not any(value.startswith("Host research context follows.") for value in contents)
    assert not any(
        value.startswith("MMM reviewed Skill/tool/Minecraft-MCP routing context:")
        for value in contents
    )
    assert "Repository branch policy: main only" in contents
    assert any("HOST FORCED ACT:" in value for value in contents)
    assert any("Use create_file" in value for value in contents)
    assert any(target in value for value in contents)


def test_output_boundary_recovery_matches_visible_source_edit_protocol() -> None:
    request = GenerationRequest(
        messages=({"role": "user", "content": "implement"},),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "description": "edit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["create_file", "create"],
                            },
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["operation", "content"],
                    },
                },
            },
        ),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
        parallel_tool_calls=False,
    )

    instruction = tool_loop._atomic_output_recovery_instruction(request)

    assert "apply_source_edit exactly once" in instruction
    assert "operation=create_file" in instruction
    assert "create_java_type" not in instruction
