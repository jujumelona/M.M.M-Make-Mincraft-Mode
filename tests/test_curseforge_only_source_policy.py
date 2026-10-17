"""The mod-source workflow must never reach GitHub after CurseForge-only policy."""

import pytest

from minecraft_mod_ai.ecosystem_discovery import (
    EcosystemDiscoveryClient, EcosystemDiscoveryUnavailable,
)
from minecraft_mod_ai.github_public_source import (
    PublicGitSourceError, inspect_public_repository, public_git_available,
)
from minecraft_mod_ai.pre_design_external_source_contract import (
    _retrieve_github_source_body, _augment_bundle,
)
from minecraft_mod_ai.reference_source_research import _github_reference_sources
from minecraft_mod_ai.reuse_discovery import discover_repositories_for_graph


def test_github_discovery_denied_before_http():
    with EcosystemDiscoveryClient() as client:
        with pytest.raises(Exception, match="Unsupported ecosystem discovery provider"):
            client.search("github", "space mod")
        with pytest.raises(EcosystemDiscoveryUnavailable, match="disabled"):
            client.inspect_github_repository("example/mod", minecraft_version="26.2", loader="fabric")


def test_public_git_fallback_forbidden():
    assert public_git_available() is False
    with pytest.raises(PublicGitSourceError, match="disabled"):
        inspect_public_repository("example/mod", "26.2", "fabric", 1000, 1000, 1000)


def test_predesign_github_retrieval_is_noop():
    receipt = _retrieve_github_source_body("space mod")
    assert receipt["search_requests"] == receipt["source_requests"] == 0
    assert receipt["records"] == []
    bundle = {"domains": [{"domain_id": "minecraft_api", "queries": [{"query": "space"}]}]}
    assert _augment_bundle({}, bundle) == bundle


def test_reference_github_retrieval_is_noop():
    records, receipt = _github_reference_sources(["space"], ["space"])
    assert records == []
    assert receipt["status"] == "disabled"


@pytest.mark.parametrize("provider", ["github", "modrinth", "sourcegraph"])
def test_reuse_only_accepts_curseforge(provider):
    with pytest.raises(ValueError, match="only permits CurseForge"):
        discover_repositories_for_graph(["space"], None, providers=(provider,))
