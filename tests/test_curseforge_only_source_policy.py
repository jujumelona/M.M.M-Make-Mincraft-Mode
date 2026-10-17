"""Regression checks for deleted external repository search implementations."""
import pytest
from minecraft_mod_ai import ecosystem_discovery, pre_design_grounded_rag
from minecraft_mod_ai import reference_source_research, reuse_discovery

def test_removed_discovery_methods_are_not_present():
    assert not hasattr(ecosystem_discovery.EcosystemDiscoveryClient, "_search_github")
    assert not hasattr(ecosystem_discovery.EcosystemDiscoveryClient, "inspect_github_repository")
    assert not hasattr(pre_design_grounded_rag, "_search_github")
    assert not hasattr(pre_design_grounded_rag, "_linked_github_sources")
    assert not hasattr(reference_source_research, "_github_reference_sources")
    assert not hasattr(reuse_discovery, "_search_sourcegraph")

@pytest.mark.parametrize("provider", ["github", "modrinth", "sourcegraph"])
def test_only_curseforge_catalogue_allowed(provider):
    with pytest.raises(ValueError, match="Only the CurseForge catalogue"):
        reuse_discovery.discover_repositories_for_graph(["space"], None, providers=(provider,))

def test_external_github_provider_not_supported():
    with ecosystem_discovery.EcosystemDiscoveryClient() as client:
        with pytest.raises(Exception, match="Unsupported ecosystem discovery provider"):
            client.search("github", "space")
