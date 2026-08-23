from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import api
from minecraft_mod_ai import complete_orchestrator, complete_planner
from minecraft_mod_ai import platform_central_ai_contract as central_contract
from minecraft_mod_ai import reuse_planner as reuse
from minecraft_mod_ai.evidence_first_planning import compile_evidence_first_plan
from minecraft_mod_ai.platform_catalog import PlatformAdapter
from minecraft_mod_ai.project_inventory import inspect_project_inventory
from minecraft_mod_ai.spec import SpecValidationError, canonical_json


def _write_existing_project(root: Path) -> None:
    (root / "src/main/java/example").mkdir(parents=True)
    (root / "src/main/resources").mkdir(parents=True)
    (root / "settings.gradle.kts").write_text(
        'rootProject.name = "weather-existing"\n', encoding="utf-8"
    )
    (root / "gradle.properties").write_text(
        "minecraft_version=1.21.1\nloader_version=0.16.10\n",
        encoding="utf-8",
    )
    (root / "src/main/java/example/WeatherCompass.java").write_text(
        "package example; public final class WeatherCompass {}\n",
        encoding="utf-8",
    )
    (root / "src/main/resources/fabric.mod.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "weather_existing",
                "version": "1.0.0",
                "depends": {"minecraft": "1.21.1"},
            }
        ),
        encoding="utf-8",
    )


def _archive_project(project: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w") as output:
        for path in sorted(project.rglob("*")):
            if path.is_file():
                output.write(path, "project/" + path.relative_to(project).as_posix())


def _retained_evidence_plan(root: Path, prompt: str) -> tuple[dict, dict]:
    _write_existing_project(root)
    inventory = inspect_project_inventory(root).to_dict()
    design = {
        "pitch": "Keep the existing weather compass behavior.",
        "modules": [{"plugin_id": "weather_compass", "reason": "weather_compass"}],
        "acceptance_tests": ["The existing weather compass remains available."],
        "_existing_project_inventory": inventory,
        "_existing_snapshot": inventory,
        "_platform_selection": {
            "target": {
                "minecraft_version": "1.21.1",
                "loader": "fabric",
                "source_api_family": "fabric_live_ai",
            },
            "preserved_existing_target": True,
            "migration_requested": False,
        },
    }
    return design, compile_evidence_first_plan(prompt, design)


def test_existing_report_inventory_and_proposal_share_one_observed_archive_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    archive = tmp_path / "existing.zip"
    _write_existing_project(project)
    _archive_project(project, archive)
    owner = SimpleNamespace()
    api._attach_existing_target(owner, archive)
    inventory = owner._mmm_existing_project_inventory_future.result(timeout=30)
    expected = owner._mmm_existing_archive_sha256

    assert owner._mmm_existing_project_report["archive_sha256"] == expected
    assert inventory.source_sha256 == expected
    assert api._verified_existing_input_sha256(
        owner, archive, await_inventory=True
    ) == expected

    captured: dict[str, str] = {}

    class Planner:
        def plan(self, prompt, *, media_paths=(), existing_input_sha256=""):
            captured["sha256"] = existing_input_sha256
            return SimpleNamespace(
                requested_prompt=prompt,
                game_design={},
                modules=(),
                acceptance_tests=(),
                existing_input_sha256=existing_input_sha256,
                calculate_hash=lambda: "sha256:" + "a" * 64,
            )

    session = object.__new__(api.CompleteModAISession)
    session.router = owner
    session.existing_input = archive
    session.planner = Planner()
    session.brief = ""
    session.complete_proposal = None
    monkeypatch.setattr(api.CompleteModAISession, "save_plan", lambda self: archive)
    import minecraft_mod_ai.plan_render as plan_render

    monkeypatch.setattr(plan_render, "render_complete_plan", lambda **_kwargs: "plan")
    reply = session.chat("Keep the existing project.")

    assert captured["sha256"] == expected
    assert reply.complete_proposal.existing_input_sha256 == expected

    archive.write_bytes(b"changed archive bytes")
    with pytest.raises(SpecValidationError, match="changed after the session observed"):
        api._verified_existing_input_sha256(owner, archive, await_inventory=True)


def test_fast_mode_does_not_reduce_model_context_or_completion_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = {
        role: SimpleNamespace(max_context=262_144, max_new_tokens=65_536)
        for role in ("planner", "coder", "researcher", "coder_safe", "visual_critic")
    }

    class Registry:
        def role(self, _profile: str, role: str):
            return configs[role]

    class Router:
        def __init__(self, *, profile: str):
            self.profile = profile
            self.registry = Registry()

    class Planner:
        def __init__(self, router):
            self.router = router

    class Orchestrator:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr("minecraft_mod_ai.model_router.ModelRouter", Router)
    monkeypatch.setattr(complete_planner, "CompleteGameDesignPlanner", Planner)
    monkeypatch.setattr(
        complete_orchestrator, "CompleteProductionOrchestrator", Orchestrator
    )

    session = api.CompleteModAISession(output_root=tmp_path, fast_mode=True)

    assert session.fast_mode is True
    assert {
        (config.max_context, config.max_new_tokens) for config in configs.values()
    } == {(262_144, 65_536)}


def test_live_lowering_preserves_validated_retain_only_plan_with_base_content(
    tmp_path: Path,
) -> None:
    prompt = "Keep the weather compass."
    design, evidence_plan = _retained_evidence_plan(tmp_path / "existing", prompt)
    result = SimpleNamespace(
        requested_prompt=prompt,
        game_design={**design, "_evidence_first_plan": evidence_plan},
        modules=(),
        base_proposal=SimpleNamespace(
            spec=SimpleNamespace(contents=(object(),), boss=None)
        ),
    )

    class Planner:
        def _plan_in_session(self, prompt, *, media_paths=(), existing_input_sha256=""):
            return result

    module = SimpleNamespace(
        CompleteGameDesignPlanner=Planner,
        SpecValidationError=SpecValidationError,
    )
    central_contract._install_live_module_lowering(module)

    assert Planner()._plan_in_session(prompt) is result
    assert result.modules == ()


def _adapter(index: int) -> PlatformAdapter:
    version = f"candidate-{index:02d}"
    return PlatformAdapter(
        adapter_id=f"fabric-{version}",
        edition="java",
        loader="fabric",
        minecraft_version=version,
        java_version="21",
        yarn_mappings=f"{version}+build.1",
        fabric_loader="0.16.10",
        fabric_api="1.0.0",
        fabric_loom="1.8",
        gradle="8.10.2",
        gradle_sha256="sha256:" + "b" * 64,
        resource_pack_format=34,
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def _research_receipt(adapter: PlatformAdapter) -> dict:
    payload = {
        "schema_version": "mmm/central-evidence-graph-v1",
        "brief_sha256": "sha256:" + "c" * 64,
        "target": {
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
        },
        "domains": [
            {
                "domain_id": "platform_target",
                "fusion": {"critic": {"mean_coverage": 1.0}},
            }
        ],
        "unresolved_official_domains": [],
        "authorization": "none",
        "retrieval_is_authority": False,
    }
    payload["evidence_sha256"] = "sha256:" + hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _configure_official_gate_fixture(
    monkeypatch: pytest.MonkeyPatch,
    adapters: tuple[PlatformAdapter, ...],
) -> None:
    by_key = {(item.loader, item.minecraft_version): item for item in adapters}

    def plan_target(adapter, **_kwargs):
        index = int(adapter.minecraft_version.rsplit("-", 1)[1])
        decision = reuse.ReuseDecision(
            capability="gameplay.core",
            mode="fresh",
            confidence=1.0,
            fresh_implementation_cost=float(index + 1),
            fresh_verification_cost=0.0,
        )
        return reuse.TargetImplementationPlan(
            adapter=adapter,
            capabilities=(decision,),
            platform_evidence=None,
            cross_component_integration_cost=0.0,
            platform_verification_cost=1.0,
            maintenance_risk=0.5,
            total_expected_cost=float(index + 2),
            weighted_verified_reuse=0.0,
            fresh_work=float(index + 1),
            adaptation_work=0.0,
            verification_work=0.0,
            uncertainty=0.0,
            reusable_registry_candidates=0,
        )

    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "on")
    monkeypatch.setattr(
        reuse,
        "discover_target_keys",
        lambda **_kwargs: tuple(by_key),
    )
    monkeypatch.setattr(
        reuse,
        "adapter_for_target",
        lambda version, loader: by_key[(loader, version)],
    )
    monkeypatch.setattr(reuse, "load_verified_components", lambda: ())
    monkeypatch.setattr(
        reuse,
        "_parallel_donor_repository_discovery",
        lambda queries, _client: {query: () for query in queries},
    )
    monkeypatch.setattr(reuse, "_plan_target", plan_target)
    monkeypatch.setattr(
        reuse._platform,
        "_parallel_support_matrix",
        lambda candidates, queries, _client: (
            {
                item.adapter_id: {query: () for query in queries}
                for item in candidates
            },
            (),
        ),
    )
    monkeypatch.setattr(
        reuse._platform,
        "_parallel_deep",
        lambda candidates, *, queries, **_kwargs: tuple(
            reuse._fresh_evidence(item, queries) for item in candidates
        ),
    )


def test_every_feasible_target_is_evidenced_and_unverified_target_cannot_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = tuple(_adapter(index) for index in range(15))
    by_key = {(item.loader, item.minecraft_version): item for item in adapters}
    observed: dict[str, object] = {}

    def discover(**kwargs):
        observed["discovery_limit"] = kwargs["limit_per_loader"]
        return tuple(by_key)

    def plan_target(adapter, **_kwargs):
        index = int(adapter.minecraft_version.rsplit("-", 1)[1])
        decision = reuse.ReuseDecision(
            capability="gameplay.core",
            mode="fresh",
            confidence=1.0,
            fresh_implementation_cost=float(index + 1),
            fresh_verification_cost=0.0,
        )
        return reuse.TargetImplementationPlan(
            adapter=adapter,
            capabilities=(decision,),
            platform_evidence=None,
            cross_component_integration_cost=0.0,
            platform_verification_cost=1.0,
            maintenance_risk=0.5,
            total_expected_cost=float(index + 2),
            weighted_verified_reuse=0.0,
            fresh_work=float(index + 1),
            adaptation_work=0.0,
            verification_work=0.0,
            uncertainty=0.0,
            reusable_registry_candidates=0,
        )

    def support_matrix(candidates, queries, _client):
        observed["matrix"] = tuple(item.adapter_id for item in candidates)
        return (
            {
                item.adapter_id: {query: () for query in queries}
                for item in candidates
            },
            (),
        )

    def deep(candidates, *, queries, **_kwargs):
        observed["deep"] = tuple(item.adapter_id for item in candidates)
        # The cheapest pre-evidence candidate deliberately fails verification.
        return tuple(
            reuse._fresh_evidence(item, queries)
            for item in candidates
            if item is not adapters[0]
        )

    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "on")
    monkeypatch.setattr(reuse, "discover_target_keys", discover)
    monkeypatch.setattr(
        reuse,
        "adapter_for_target",
        lambda version, loader: by_key[(loader, version)],
    )
    monkeypatch.setattr(reuse, "load_verified_components", lambda: ())
    monkeypatch.setattr(
        reuse,
        "_parallel_donor_repository_discovery",
        lambda queries, _client: {query: () for query in queries},
    )
    monkeypatch.setattr(reuse, "_plan_target", plan_target)
    monkeypatch.setattr(reuse._platform, "_parallel_support_matrix", support_matrix)
    monkeypatch.setattr(reuse._platform, "_parallel_deep", deep)

    optimized = reuse.optimize_platform_and_reuse(
        "Implement gameplay core.", discovery_client=object()
    )

    expected_ids = tuple(item.adapter.adapter_id for item in optimized.candidate_plans)
    assert set(observed["matrix"]) == {item.adapter_id for item in adapters}
    assert set(observed["deep"]) == {item.adapter_id for item in adapters}
    assert observed["discovery_limit"] != 12
    assert optimized.selected.adapter_id == adapters[1].adapter_id
    assert adapters[0].adapter_id not in expected_ids
    assert all(item.platform_evidence is not None for item in optimized.candidate_plans)


def test_unavailable_winner_falls_back_to_next_valid_official_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = tuple(_adapter(index) for index in range(3))
    _configure_official_gate_fixture(monkeypatch, adapters)
    calls: list[str] = []

    def research(adapter: PlatformAdapter):
        calls.append(adapter.adapter_id)
        if adapter is adapters[0]:
            return {
                "schema_version": "mmm/central-evidence-graph-v1",
                "status": "unavailable",
                "target": {
                    "minecraft_version": adapter.minecraft_version,
                    "loader": adapter.loader,
                    "mappings": adapter.yarn_mappings,
                },
                "domains": [],
                "unresolved_official_domains": ["platform_target"],
                "authorization": "none",
                "retrieval_is_authority": False,
            }
        return _research_receipt(adapter)

    optimized = reuse.optimize_platform_and_reuse(
        "Implement gameplay core.",
        discovery_client=object(),
        target_research_fn=research,
    )

    assert calls == [adapters[0].adapter_id, adapters[1].adapter_id]
    assert optimized.selected.adapter_id == adapters[1].adapter_id
    assert adapters[0].adapter_id not in {
        item.adapter.adapter_id for item in optimized.candidate_plans
    }
    assert optimized.selected_plan.platform_evidence.deep_research == _research_receipt(
        adapters[1]
    )


def test_all_invalid_official_target_receipts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = tuple(_adapter(index) for index in range(3))
    _configure_official_gate_fixture(monkeypatch, adapters)
    calls: list[str] = []

    def research(adapter: PlatformAdapter):
        calls.append(adapter.adapter_id)
        return {
            "schema_version": "mmm/central-evidence-graph-v1",
            "status": "unavailable",
            "target": {
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "mappings": adapter.yarn_mappings,
            },
            "domains": [],
            "unresolved_official_domains": ["platform_target"],
            "authorization": "none",
            "retrieval_is_authority": False,
        }

    with pytest.raises(ValueError, match="valid target research evidence"):
        reuse.optimize_platform_and_reuse(
            "Implement gameplay core.",
            discovery_client=object(),
            target_research_fn=research,
        )

    assert calls == [item.adapter_id for item in adapters]
