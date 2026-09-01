from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import platform_selection_pipeline as strict_platform
from minecraft_mod_ai.planning_pipeline import PlanningPipeline, PlanningStageError
from minecraft_mod_ai.platform_catalog import PlatformAdapter, PlatformProvider
from minecraft_mod_ai.spec import PlatformLock, SpecValidationError

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"


def _adapter(version: str) -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id=f"fixture_{version.replace('.', '_')}",
        edition="java",
        loader="fabric",
        minecraft_version=version,
        java_version="21",
        yarn_mappings="mojang",
        mappings_kind="mojang",
        mappings_version="mojang",
        fabric_loader="0.16.0",
        fabric_api="1.0.0",
        fabric_loom="1.9",
        gradle="8.10",
        gradle_sha256="0" * 64,
        data_pack_version="61",
        resource_pack_version="46",
        resource_pack_format=46,
        release_metadata_url="https://piston-meta.mojang.com/fixture",
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def test_complete_planner_uses_canonical_pipeline_not_legacy_game_design_planner() -> None:
    source = (PACKAGE / "complete_planner.py").read_text(encoding="utf-8")
    assert "PlanningPipeline(self.router).prepare" in source
    assert "GameDesignPlanner(self.router).plan" not in source
    assert "research-unavailable" not in source


def test_unavailable_evidence_cannot_advance() -> None:
    with pytest.raises(PlanningStageError, match="explicitly unavailable"):
        PlanningPipeline._validated_evidence(
            {
                "schema_version": "mmm/central-evidence-graph-v1",
                "status": "unavailable",
            }
        )


def test_platform_lock_validation_is_offline_and_does_not_reresolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import minecraft_mod_ai.platform_catalog as catalog

    def forbidden(*_args, **_kwargs):
        raise AssertionError("proposal validation must not resolve a provider")

    monkeypatch.setattr(catalog, "adapter_for_lock_values", forbidden)
    lock = PlatformLock(
        edition="java",
        loader="fabric",
        minecraft_version="1.21.1",
        java_version="21",
        yarn_mappings="mojang",
        fabric_loader="0.16.0",
        fabric_api="1.0.0+1.21.1",
        fabric_loom="1.9",
        gradle="8.10",
    )
    lock.validate()


def test_candidate_receipt_is_resolved_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def resolve(version: str) -> PlatformAdapter:
        calls.append(version)
        return _adapter(version)

    provider = PlatformProvider(
        loader="fabric",
        provider_id="fixture",
        discover_versions=lambda limit: ("1.21.1", "1.21.2")[:limit],
        resolve=resolve,
    )
    monkeypatch.setattr(strict_platform, "executable_loaders", lambda: ("fabric",))
    monkeypatch.setattr(strict_platform, "provider_for_loader", lambda _loader: provider)

    adapters = strict_platform._resolved_candidates(
        loader_constraint=None,
        version_constraint=None,
        page_size=2,
    )

    assert [item.minecraft_version for item in adapters] == ["1.21.1", "1.21.2"]
    assert calls == ["1.21.1", "1.21.2"]


def test_deep_evidence_failure_is_not_converted_to_fresh_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter("1.21.1")

    def fail(*_args, **_kwargs):
        raise RuntimeError("inspection backend unavailable")

    monkeypatch.setattr(strict_platform.legacy_optimizer, "_deep_evidence", fail)
    with pytest.raises(SpecValidationError, match="Refusing fresh-only fallback"):
        strict_platform._parallel_deep_fail_closed(
            (adapter,),
            queries=("gameplay.core",),
            matrix={adapter.adapter_id: {}},
            client=object(),
            target_research_fn=None,
            inherited_errors=(),
            shallow_candidate_count=0,
        )


def test_strict_platform_path_contains_no_post_admission_skip_or_rediscovery() -> None:
    source = (PACKAGE / "platform_selection_pipeline.py").read_text(encoding="utf-8")
    assert "discover_target_keys" not in source
    assert "adapter_for_target" not in source
    assert "target resolution skipped" not in source
    assert "using fresh-only evidence" not in source
