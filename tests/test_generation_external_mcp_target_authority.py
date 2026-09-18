from __future__ import annotations

from jsonschema import Draft202012Validator

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime
from minecraft_mod_ai.external_agent_bridge import ExternalAgentBridge


def _schema(name: str) -> dict:
    return next(
        item["function"]["parameters"]
        for item in ExternalAgentBridge.tool_schemas("generation")
        if item["function"]["name"] == name
    )


def test_generation_capability_schema_accepts_model_version_scalar_before_host_binding() -> None:
    schema = _schema("external_mcp_capabilities")

    # Regression for the Debug Mode trace where the small model emitted 26.2 as
    # a JSON number. Generation target coordinates are host-owned, so their
    # model representation must not reject the call before host normalization.
    Draft202012Validator(schema).validate(
        {
            "minecraft_version": 26.2,
            "loader": "fabric",
            "mappings": "official",
            "max_access": "read",
        }
    )


def test_generation_external_mcp_call_uses_project_platform_lock(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "build.gradle").write_text("// fixture\n", encoding="utf-8")
    metadata = tmp_path / ".minecraft_ai"
    metadata.mkdir()
    (metadata / "platform-lock.json").write_text(
        """{
  "minecraft_version": "26.2",
  "loader": "fabric",
  "mappings_kind": "official",
  "mappings_version": "",
  "yarn_mappings": ""
}
""",
        encoding="utf-8",
    )

    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    captured: dict[str, object] = {}

    def fake_call(stage, name, payload, *, allowed_server_ids=None):
        captured["stage"] = stage
        captured["name"] = name
        captured["payload"] = dict(payload)
        captured["allowed_server_ids"] = allowed_server_ids
        return {"status": "PASS"}

    monkeypatch.setattr(runtime._external_bridge, "call", fake_call)

    runtime.call(
        "generation",
        "external_mcp_capabilities",
        {
            "minecraft_version": 1.20,
            "loader": "forge",
            "mappings": "yarn",
            "max_access": "admin",
        },
    )

    assert captured["stage"] == "generation"
    assert captured["name"] == "external_mcp_capabilities"
    assert captured["payload"] == {
        "minecraft_version": "26.2",
        "loader": "fabric",
        "mappings": "official",
        "max_access": "read",
    }


def test_repeated_rejected_evidence_route_advances_frontier() -> None:
    state = tool_loop.HostRunState(phase=tool_loop.LoopPhase.OBSERVE)
    state.semantic_fixed_point = True
    state.no_progress_streak = 1
    state.seen_no_progress_digests.add("same-state")

    routes = tool_loop._consume_rejected_evidence_fixed_point(
        state,
        (
            {
                "failure_code": "TOOL_SCHEMA_INVALID",
                "original_tool": "external_mcp_capabilities",
            },
        ),
        {"external_mcp_capabilities"},
    )

    assert routes == ("external_mcp_capabilities",)
    assert "external_mcp_capabilities" in state.attempted_sources
    assert state.semantic_fixed_point is False
    assert state.no_progress_streak == 0
