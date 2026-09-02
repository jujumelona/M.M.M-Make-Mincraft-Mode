from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import platform_evidence_pipeline as evidence
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


def test_deleted_legacy_planner_files_are_physically_absent() -> None:
    assert not (PACKAGE / "platform_optimizer.py").exists()
    assert not (PACKAGE / "platform_central_ai_contract.py").exists()
    assert not (ROOT / "tests" / "test_platform_failure_recovery.py").exists()


def test_complete_planner_uses_canonical_pipeline_and_no_synthetic_module_fallback() -> None:
    source = (PACKAGE / "complete_planner.py").read_text(encoding="utf-8")
    assert "PlanningPipeline(self.router).prepare" in source
    assert "GameDesignPlanner(self.router).plan" not in source
    assert "research-unavailable" not in source
    assert "Implement the complete requested mod behavior" not in source
    assert "from .live_module_lowering import lower_live_modules" in source


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


def test_capability_decomposition_does_not_semantically_truncate_at_twelve() -> None:
    design = {
        "features": [
            {"name": f"capability_{index}"}
            for index in range(30)
        ]
    }
    queries = evidence.capability_queries("make the requested mod", design=design)
    assert len(queries) == 30
    assert queries[-1] == "capability 29"


def test_frontier_provider_receipt_is_resolved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def resolve(version: str) -> PlatformAdapter:
        calls.append(version)
        return _adapter(version)

    provider = PlatformProvider(
        loader="fabric",
        provider_id="fixture",
        discover_versions=lambda _limit: (),
        resolve=resolve,
    )
    monkeypatch.setattr(evidence, "provider_for_loader", lambda _loader: provider)

    adapters, errors = evidence._resolve_frontier(
        (("fabric", "1.21.1"), ("fabric", "1.21.1"), ("fabric", "1.21.2"))
    )
    assert errors == ()
    assert [item.minecraft_version for item in adapters] == ["1.21.1", "1.21.2"]
    assert calls == ["1.21.1", "1.21.2"]


def test_target_research_failure_cannot_be_ranked_as_fresh_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PlatformProvider(
        loader="fabric",
        provider_id="fixture",
        discover_versions=lambda _limit: (),
        resolve=lambda version: _adapter(version),
    )
    monkeypatch.setattr(evidence, "provider_for_loader", lambda _loader: provider)
    monkeypatch.setattr(evidence, "executable_loaders", lambda: ("fabric",))

    def fail(_adapter: PlatformAdapter):
        raise RuntimeError("research backend unavailable")

    with pytest.raises(SpecValidationError, match="failed closed"):
        evidence.optimize_platform_evidence(
            "Minecraft 1.21.1 Fabric test mod",
            version_constraint="1.21.1",
            loader_constraint="fabric",
            target_research_fn=fail,
        )


def test_canonical_platform_sources_contain_no_legacy_recovery_or_optimizer_import() -> None:
    selection = (PACKAGE / "platform_selection_pipeline.py").read_text(encoding="utf-8")
    evidence_source = (PACKAGE / "platform_evidence_pipeline.py").read_text(encoding="utf-8")
    resolver = (PACKAGE / "platform_resolver.py").read_text(encoding="utf-8")
    combined = "\n".join((selection, evidence_source, resolver))
    assert "platform_optimizer" not in combined
    assert "target resolution skipped" not in combined
    assert "using fresh-only evidence" not in combined
    assert "_DEPENDENCY_NODE_BUDGET" not in combined
    assert "root_ids[:" not in combined
    assert "top_k" not in combined
