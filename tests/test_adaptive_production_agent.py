from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.production_tools import ProductionToolService
from minecraft_mod_ai.skill_catalog import REVIEWED_TOOL_STAGES


class _Registry:
    def __init__(self) -> None:
        self.config = SimpleNamespace(adapter="llama_cpp", exclusive_gpu=False)

    def load_profile(self, profile: str) -> None:
        assert profile == "test"

    def role(self, profile: str, role: str):
        return self.config


def _schema(name: str):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_bound_production_coder_cannot_finalize_before_fresh_rag(monkeypatch, tmp_path: Path) -> None:
    class Runtime:
        def tool_schemas(self, stage):
            return (_schema("search_code_rag"),)

        def call(self, stage, name, arguments):
            return {
                "hits": [{"path": "src/main/java/X.java"}],
                "receipt": {
                    "result_count": 1,
                    "coverage_score": 1.0,
                    "relevance_score": 1.0,
                },
            }

    class Adapter:
        def __init__(self):
            self.count = 0

        def generate_turn(self, request):
            self.count += 1
            if self.count == 1:
                return GenerationResponse(content="premature")
            if self.count == 2:
                return GenerationResponse(tool_calls=(ToolCall(
                    id="rag",
                    name="search_code_rag",
                    arguments={"query": "Registry.register"},
                    raw_arguments='{"query":"Registry.register"}',
                ),))
            return GenerationResponse(content="evidence-backed final")

    adapter = Adapter()
    runtime = Runtime()
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
    router.bind_agent_workspace(tmp_path, require_fresh_evidence=True)
    assert router.generate_text(
        "coder",
        [{"role": "user", "content": "implement"}],
    ) == "evidence-backed final"
    assert adapter.count == 3


def test_independent_read_tools_execute_in_parallel(monkeypatch) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    # Both reads are equal-cost first steps for the observe frontier. This proves the
    # executor still overlaps independent reads without requiring the causal planner
    # to expose a redundant, non-minimal retrieval route merely for the test.
    class Runtime:
        def tool_schemas(self, stage):
            return (_schema("search_code_rag"), _schema("java_workspace_symbols"))

        def call(self, stage, name, arguments):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.08)
            with lock:
                active -= 1
            return {"evidence": name}

    class Adapter:
        def __init__(self):
            self.count = 0

        def generate_turn(self, request):
            self.count += 1
            if self.count == 1:
                assert {
                    item["function"]["name"] for item in request.tools
                } == {"search_code_rag", "java_workspace_symbols"}
                return GenerationResponse(tool_calls=(
                    ToolCall(
                        id="a",
                        name="search_code_rag",
                        arguments={"query": "a"},
                        raw_arguments='{"query":"a"}',
                    ),
                    ToolCall(
                        id="b",
                        name="java_workspace_symbols",
                        arguments={"query": "b"},
                        raw_arguments='{"query":"b"}',
                    ),
                ))
            return GenerationResponse(content="done")

    adapter = Adapter()
    runtime = Runtime()
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
    assert router.generate_text(
        "coder",
        [{"role": "user", "content": "x"}],
    ) == "done"
    assert max_active == 2


def test_generation_stage_exposes_required_evidence_and_quality_stays_narrow() -> None:
    generation = {
        "search_project_rag",
        "search_code_rag",
        "discover_ecosystem_resources",
        "inspect_github_repository",
        "inspect_modrinth_project",
        "java_diagnostics",
        "java_workspace_symbols",
    }
    for name in generation:
        assert "generation" in REVIEWED_TOOL_STAGES[name]
    assert "quality" not in REVIEWED_TOOL_STAGES["inspect_github_repository"]
    assert "quality" not in REVIEWED_TOOL_STAGES["discover_ecosystem_resources"]
    assert "quality" in REVIEWED_TOOL_STAGES["search_project_rag"]


def test_code_rag_index_can_be_refreshed_in_place(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src/main/java/example/X.java"
    source.parent.mkdir(parents=True)
    source.write_text("class X { int oldValue; }", encoding="utf-8")
    service = ProductionToolService(workspace_root=tmp_path, profile="test")
    metadata = {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mapping_namespace": "yarn",
        "java_version": "17",
        "license": "project-local",
        "source_commit": "first",
    }
    service.index_project_rag(["project"], metadata=metadata)
    source.write_text("class X { int newValue; }", encoding="utf-8")
    metadata["source_commit"] = "second"
    service.index_project_rag(["project"], metadata=metadata)
    result = service.search_code_rag("newValue")
    assert result["hits"]


def test_reviewed_stage_map_matches_live_mcp_map(monkeypatch) -> None:
    monkeypatch.setenv("MMM_MCP_STAGE", "all")
    from minecraft_mod_ai import mcp_server

    for name, stages in REVIEWED_TOOL_STAGES.items():
        live = mcp_server._TOOL_STAGES[name]
        if name == "discover_mmm_capabilities":
            assert live - {"all"} == stages
        else:
            assert live == stages
