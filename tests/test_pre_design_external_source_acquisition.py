"""Retired external repository research leaves grounded catalogue data untouched."""
from types import SimpleNamespace

from minecraft_mod_ai import pre_design_external_source_contract as external

def test_legacy_external_search_cannot_issue_requests():
    result = external._retrieve_github_source_body("spaceflight")
    assert result["records"] == []
    assert result["search_requests"] == result["source_requests"] == 0

def test_local_grounded_bundle_is_preserved():
    bundle = {"domains": [{"queries": [{"query": "spaceflight", "external_rag": {"records": [{"source_id": "curseforge:1"}]}}]}]}
    assert external._augment_bundle({}, bundle) == bundle

def test_retired_install_does_not_wrap_research():
    original = lambda _router, _request: {"domains": [{"queries": []}]}
    mod = SimpleNamespace(_forced_rag_bundle=original)
    external.install(mod)
    assert mod._forced_rag_bundle is original
