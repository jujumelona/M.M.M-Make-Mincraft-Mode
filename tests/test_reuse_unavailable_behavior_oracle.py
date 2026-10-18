"""Do not run expensive Gradle on a donor that cannot pass any bound behavior test."""
from types import SimpleNamespace

from minecraft_mod_ai import platform_catalog
from minecraft_mod_ai.curseforge_reuse_production import certify_sources_at_production


def test_missing_implementation_bound_contract_skips_futile_gradle(monkeypatch, tmp_path):
    monkeypatch.setattr(
        platform_catalog, "adapter_for_target",
        lambda _version, _loader: SimpleNamespace(validate=lambda: None),
    )
    source = {"capability": "uncatalogued.teleport",
              "source_id": "curseforge:4",
              "source_url": "https://github.com/example/mod",
              "reference_only": True, "proof_status": "NOT_RUN",
              "semantic_intent": "reuse", "semantic_score": 94,
              "semantic_scores": {"functional_fit": 5, "completeness": 4}}
    plan = {
        "schema_version": "mmm/grounded-repository-reuse-plan-v2",
        "bound_target": {"minecraft_version": "1.21.1", "loader": "fabric"},
        "capabilities": [{"capability": "uncatalogued.teleport", "mode": "fresh"}],
        "reference_sources": [source], "proof_receipts": [],
    }
    def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("Should not fetch donor bytes before a behavior oracle exists")
    result = certify_sources_at_production(
        plan, project_root=tmp_path,
        target_context={"minecraft_version": "1.21.1", "loader": "fabric"},
        source_fetcher=unexpected_fetch,
    )
    assert result["proof_attempt_count"] == 0
    assert result["capabilities"][0]["mode"] == "fresh"
    assert result["proof_receipts"][0]["status"] == "behavior_contract_unavailable"
    assert result["proof_receipts"][0]["failure_code"] == "NO_IMPLEMENTATION_BOUND_ACCEPTANCE"
    assert result["reference_sources"] == [source]
