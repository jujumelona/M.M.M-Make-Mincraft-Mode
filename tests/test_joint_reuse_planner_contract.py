from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import component_registry
from minecraft_mod_ai.project_inventory import inspect_project_inventory
from minecraft_mod_ai.reuse_planner import (
    ReuseDecision,
    _declared_same_project_capabilities,
    _plan_target,
    decompose_capability_graph,
)
from minecraft_mod_ai.source_transplant import _target_compatibility


def test_capability_graph_uses_behavior_not_whole_mod_theme() -> None:
    graph = decompose_capability_graph(
        "메이플스토리 같은 모드",
        design={"systems": [{"id": "trade"}, {"id": "shop"}]},
    )
    assert "trade.transaction" in graph.nodes
    assert "trade.validation" in graph.nodes
    assert "ui.shop_menu" in graph.nodes
    assert all("maple" not in item and "메이플" not in item for item in graph.nodes)


def test_capability_graph_merges_prompt_requirements_missing_from_design() -> None:
    graph = decompose_capability_graph(
        "Add trade and quests.",
        design={"systems": [{"id": "trade"}]},
    )

    assert "trade.transaction" in graph.nodes
    assert "quest.state" in graph.nodes
    assert "quest.progression" in graph.nodes
    assert "quest.reward" in graph.nodes


def test_evidence_catalog_capability_is_not_prefixed_or_rewritten() -> None:
    capability = "interdimensional_player_owned_energy_distribution_network_audited_access"
    prompt = capability.replace("_", " ") + "."
    catalog = {
        "requirements": [
            {
                "requirement_id": "req-custom",
                "capability": capability,
                "statement": prompt,
                "provides": [f"capability:{capability}"],
            }
        ]
    }

    graph = decompose_capability_graph(
        prompt,
        design={"_evidence_request_catalog": catalog},
    )

    assert graph.nodes == (capability,)


def test_evidence_catalog_does_not_suppress_uncovered_prompt_requirement() -> None:
    graph = decompose_capability_graph(
        "Add trade and quests.",
        design={
            "_evidence_request_catalog": {
                "requirements": [
                    {
                        "requirement_id": "req-trade",
                        "capability": "trade.transaction",
                        "statement": "trade",
                        "provides": ["capability:trade.transaction"],
                    }
                ]
            }
        },
    )

    assert "trade.transaction" in graph.nodes
    assert "quest.state" in graph.nodes
    assert "quest.progression" in graph.nodes
    assert "quest.reward" in graph.nodes


def test_underscore_structured_capability_is_not_ontology_rewritten() -> None:
    capability = "interdimensional_player_owned_energy_distribution_network"

    graph = decompose_capability_graph(
        capability.replace("_", " "),
        design={"systems": [{"id": capability}]},
    )

    assert graph.nodes == (capability,)


def test_capability_graph_has_no_default_logical_project_size_cap() -> None:
    systems = [{"id": f"feature.system_{index:03d}"} for index in range(96)]
    graph = decompose_capability_graph(
        "Implement the declared systems.",
        design={"systems": systems},
    )
    assert graph.nodes == tuple(
        f"feature.system_{index:03d}" for index in range(96)
    )


def test_design_claim_alone_cannot_admit_same_project_reuse() -> None:
    assert _declared_same_project_capabilities(
        {
            "existing_capabilities": ["weather_compass"],
            "same_project_capabilities": ["network.sync"],
        }
    ) == set()


def test_validated_project_receipt_admits_exact_same_project_capability(
    tmp_path: Path,
) -> None:
    (tmp_path / "settings.gradle.kts").write_text(
        'rootProject.name = "reuse-evidence"\n',
        encoding="utf-8",
    )
    source = tmp_path / "src/main/java/example/WeatherCompass.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example; public final class WeatherCompass {}\n",
        encoding="utf-8",
    )
    inventory = inspect_project_inventory(tmp_path).to_dict()

    capabilities = _declared_same_project_capabilities(
        {"_existing_project_inventory": inventory}
    )

    assert "weather_compass" in capabilities


def test_planner_materializes_same_project_reuse_as_a_proof_bound_bundle(
    tmp_path: Path,
) -> None:
    (tmp_path / "settings.gradle").write_text("rootProject.name = 'reuse'\n", encoding="utf-8")
    source = tmp_path / "src/main/java/example/WeatherCompass.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example; public final class WeatherCompass {}\n",
        encoding="utf-8",
    )
    inventory = inspect_project_inventory(tmp_path).to_dict()

    from minecraft_mod_ai.platform_catalog import adapter_for_target

    plan = _plan_target(
        adapter_for_target("1.21.1", "fabric"),
        capabilities=("weather_compass",),
        design={"_existing_project_inventory": inventory},
        platform_evidence=None,
        registry=(),
        same_project={"weather_compass"},
        discovery_client=None,
        allow_network=False,
    )

    decision = plan.capabilities[0]
    assert decision.donor_slice is None
    assert decision.artifact_bundle is not None
    assert decision.artifact_bundle.proof_receipt["proof_level"] == "HOST_VERIFIED"
    assert plan.selected_composition is not None
    assert plan.selected_composition.bundles == (decision.artifact_bundle,)


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
