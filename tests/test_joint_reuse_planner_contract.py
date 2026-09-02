from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import component_registry
from minecraft_mod_ai.reuse_planner import ReuseDecision, decompose_capability_graph
from minecraft_mod_ai.source_transplant import _target_compatibility


def _catalog(*capabilities: str) -> dict:
    return {
        "requirements": [
            {
                "requirement_id": f"req-{index:03d}",
                "capability": capability,
                "provides": [f"capability:{capability}"],
                "statement": capability.replace(".", " "),
                "depends_on": [],
            }
            for index, capability in enumerate(capabilities)
        ]
    }


def test_approved_request_catalog_is_the_capability_authority() -> None:
    design = {
        "_evidence_request_catalog": _catalog(
            "trade.transaction",
            "quest.state",
        )
    }
    graph = decompose_capability_graph(
        "Prompt contains unrelated words that must not enlarge semantic scope.",
        design=design,
    )
    assert graph.nodes == ("trade.transaction", "quest.state")
    assert all(source.startswith("request_catalog.") for _, source in graph.sources)


def test_raw_prompt_without_approved_catalog_fails_closed() -> None:
    with pytest.raises(ValueError, match="approved request catalog or explicit module kinds"):
        decompose_capability_graph("Infer a whole mod from this raw sentence")


def test_explicit_module_kinds_are_lossless_without_prompt_inference() -> None:
    graph = decompose_capability_graph(
        "ignore this prompt",
        module_kinds=("feature.alpha", "feature.beta"),
    )
    assert graph.nodes == ("feature.alpha", "feature.beta")
    assert graph.edges == ()


def test_capability_graph_has_no_default_logical_project_size_cap() -> None:
    capabilities = tuple(f"feature.system_{index:03d}" for index in range(96))
    graph = decompose_capability_graph(
        "Implement the approved requirements.",
        design={"_evidence_request_catalog": _catalog(*capabilities)},
    )
    assert graph.nodes == capabilities


def test_reuse_value_is_saved_work_not_reuse_count() -> None:
    expensive = ReuseDecision(
        capability="trade.transaction",
        mode="source_transplant",
        confidence=0.95,
        fresh_implementation_cost=40.0,
        fresh_verification_cost=15.0,
        adaptation_cost=5.0,
        integration_cost=3.0,
        dependency_cost=2.0,
        reuse_verification_cost=4.0,
        uncertainty_penalty=1.0,
    )
    trivial_a = ReuseDecision(
        capability="lang.entry",
        mode="library",
        confidence=1.0,
        fresh_implementation_cost=2.0,
        fresh_verification_cost=1.0,
        integration_cost=1.5,
        reuse_verification_cost=0.5,
    )
    trivial_b = ReuseDecision(
        capability="tag.entry",
        mode="library",
        confidence=1.0,
        fresh_implementation_cost=2.0,
        fresh_verification_cost=1.0,
        integration_cost=1.5,
        reuse_verification_cost=0.5,
    )
    assert expensive.actual_reuse_gain > trivial_a.actual_reuse_gain + trivial_b.actual_reuse_gain


def test_missing_remote_reuse_manifest_is_normal_zero_candidates(monkeypatch) -> None:
    monkeypatch.setattr(component_registry, "_read_remote_manifest", lambda *_args: None)
    assert component_registry.load_verified_components() == ()


def test_exact_transplant_target_requires_version_and_loader_evidence() -> None:
    adapter = SimpleNamespace(minecraft_version="1.20.1", loader="fabric")
    assert _target_compatibility(
        "minecraft_version=1.20.1\ndepends fabricloader >=0.15",
        adapter=adapter,
    ) == "exact"
    assert _target_compatibility("minecraft_version=1.20.1", adapter=adapter) == "adapt"
