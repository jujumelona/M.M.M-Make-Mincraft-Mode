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
