from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.repository_explorer import RepositoryExplorer


class _NoOptionalRetrievalRouter:
    pass


def test_repository_explorer_skips_missing_optional_router_capabilities(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    java = root / "src/main/java/example"
    java.mkdir(parents=True)
    (java / "Network.java").write_text(
        "package example; public final class Network { public static void registerPacket(){ Registry.register(); } }\n",
        encoding="utf-8",
    )
    result = RepositoryExplorer(
        ProjectIndex(root),
        router=_NoOptionalRetrievalRouter(),
    ).explore("which API should register packet callback", line_budget=20)
    assert result.regions
    assert result.semantic_used is False
    assert result.rerank_used is False


def test_auto_test_time_scaling_respects_native_parallel_budget(monkeypatch) -> None:
    from minecraft_mod_ai import inference_time_scaling

    monkeypatch.setenv("MMM_TEST_TIME_SCALING", "auto")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert inference_time_scaling._scaling_mode() == "off"

    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    assert inference_time_scaling._scaling_mode() == "auto"

    monkeypatch.setenv("MMM_TEST_TIME_SCALING", "on")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert inference_time_scaling._scaling_mode() == "on"


def test_auto_repair_width_keeps_single_decode_when_one_slot(monkeypatch) -> None:
    from minecraft_mod_ai import agentic_optimization_contract as agentic

    monkeypatch.setenv("MMM_TEST_TIME_SCALING", "auto")
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    engine = SimpleNamespace(_signature=lambda _evidence: "same-signature")
    evidence = {
        "diagnostics": {},
        "build": {"status": "FAIL", "error": "x" * 200},
    }
    assert agentic._repair_candidate_count(engine, evidence, ()) == 1


def test_research_symbol_filter_drops_zero_score_global_seeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from minecraft_mod_ai import research_code_context as research

    root = tmp_path / "mod"
    java = root / "src/main/java/example"
    java.mkdir(parents=True)
    (java / "Entry.java").write_text(
        "package example; public final class Entry { public void tick(){ Service.compute(); } }\n",
        encoding="utf-8",
    )
    (java / "Service.java").write_text(
        "package example; public final class Service { public static void compute(){} public void unrelated(){} }\n",
        encoding="utf-8",
    )

    class Router:
        def rerank(self, query, documents):
            return [1.0 if " tick" in (" " + document.casefold()) else 0.0 for document in documents]

    monkeypatch.setattr(
        research,
        "adapter_for_target",
        lambda _version, _loader: SimpleNamespace(
            loader="fabric",
            fabric_loader="test-loader",
            fabric_api="test-api",
            yarn_mappings="test",
            fabric_loom="test-loom",
        ),
    )
    module = SimpleNamespace(
        kind="custom_java",
        config={"feature": "tick"},
        depends_on=(),
        required_gates=(),
    )
    context = research.ResearchCodeContext(
        root,
        project_index=ProjectIndex(root),
        router=Router(),
        module=module,
        minecraft_version="test",
        loader="fabric",
        mappings="test",
        byte_budget=8192,
    )
    entries = context._entry_points("tick")
    assert entries
    assert {item.name for item in entries} == {"tick"}


def test_research_metric_vector_always_preserves_plan_alignment() -> None:
    from minecraft_mod_ai import research_code_context as research

    quality = research._quality(
        "public int compute(){ return normalize(1); }",
        path="src/main/java/example/Service.java",
    )
    metrics = research._retrieval_metrics(
        "Service compute dependency API validate",
        "public int compute(){ return normalize(1); }",
        path="src/main/java/example/Service.java",
        symbols=("compute",),
        graph_hop=1,
        quality=quality,
        target_plan="locate contract -> call normalize -> validate",
        example_plan="call normalize",
    )
    assert "plan_alignment" in metrics
    weights = research._adaptive_weights("Service.compute dependency API", metrics)
    assert "plan_alignment" in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9
