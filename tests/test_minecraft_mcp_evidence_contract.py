from __future__ import annotations

from minecraft_mod_ai import minecraft_mcp_evidence_contract as mcp_contract
from minecraft_mod_ai.central_research import (
    normalize_research_brief,
    retrieve_domain_evidence,
)
from minecraft_mod_ai.minecraft_mcp_evidence_contract import (
    collect_external_minecraft_evidence,
)
from minecraft_mod_ai.skill_catalog import compile_skill_contract

_EXTERNAL_AGENT_TOOLS = {
    "external_mcp_capabilities",
    "external_mcp_schema",
    "external_mcp_call",
}


class _FakeRouter:
    def __init__(self) -> None:
        self.requests = []

    def invoke_many(self, requests, *, max_workers=None):
        del max_workers
        self.requests = list(requests)
        return tuple(
            {
                "capability": request["capability"],
                "status": "PASS",
                "bundle_sha256": "sha256:bundle",
                "attempts": [],
                "evidence": [
                    {
                        "server": sorted(request["allowed_server_ids"])[0],
                        "tool": "search",
                        "trust": "test",
                        "arguments_sha256": "sha256:args",
                        "result_sha256": "sha256:result",
                        "result": {"hits": [request["capability"]]},
                    }
                ],
            }
            for request in self.requests
        )


def test_minecraft_evidence_skill_keeps_external_mcp_out_of_skill_allowlist() -> None:
    """External MCP is role-scoped evidence, not a canonical Skill authorization."""
    contract = compile_skill_contract("gather-adaptive-minecraft-evidence")
    assert _EXTERNAL_AGENT_TOOLS.isdisjoint(contract.allowed_tools)
    assert {
        "search_project_rag",
        "index_project_rag",
        "search_code_rag",
        "inspect_existing_mod",
    }.issubset(contract.allowed_tools)


def test_minecraft_technical_domains_gain_external_mcp_route() -> None:
    brief = normalize_research_brief("Add a custom Fabric entity", {"title": "x"})
    request = next(
        domain for domain in brief["domains"] if domain["domain_id"] == "request"
    )
    assert "external_mcp" in request["providers"]


def test_official_research_does_not_eagerly_invoke_external_mcp(monkeypatch) -> None:
    """Optional MCP evidence must not own the official/project RAG critical path."""

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("external MCP eager sweep entered official RAG")

    monkeypatch.setattr(
        mcp_contract,
        "collect_external_minecraft_evidence",
        fail_if_called,
    )
    brief = normalize_research_brief("Add a custom Fabric entity", {"title": "x"})
    result = retrieve_domain_evidence(brief)
    assert isinstance(result, dict)
    assert result.get("domains")


def test_external_evidence_is_batched_scoped_and_compact() -> None:
    brief = normalize_research_brief("Add a custom Fabric entity", {"title": "x"})
    router = _FakeRouter()
    result = collect_external_minecraft_evidence(brief, router=router)

    assert result["execution"]["parallel"] is True
    assert result["execution"]["completed_read_cache"] is True
    assert result["execution"]["single_flight_wait"] is False
    assert result["execution"]["planning_critical_path"] is False
    assert router.requests
    scopes = {tuple(sorted(row["allowed_server_ids"])) for row in router.requests}
    assert ("mcmodding-docs",) in scopes
    assert ("minecraft-dev",) in scopes
    assert all(row["stage"] == "research" for row in router.requests)
    assert all(row["max_access"] == "read" for row in router.requests)

    request_domain = next(
        row for row in result["domains"] if row["domain_id"] == "request"
    )
    evidence = request_domain["queries"][0]["capabilities"][0]["evidence"][0]
    assert "result" not in evidence
    assert evidence["result_excerpt"]
    assert evidence["result_sha256"] == "sha256:result"


def test_explicit_external_batch_deduplicates_identical_provider_calls() -> None:
    brief = {
        "brief_sha256": "sha256:test",
        "domains": [
            {
                "domain_id": "one",
                "providers": ["external_mcp"],
                "evidence_kinds": ["minecraft_api"],
                "queries": ["same exact query"],
            },
            {
                "domain_id": "two",
                "providers": ["external_mcp"],
                "evidence_kinds": ["minecraft_api"],
                "queries": ["same exact query"],
            },
        ],
    }
    router = _FakeRouter()
    result = collect_external_minecraft_evidence(brief, router=router)

    assert result["execution"]["request_count"] == 4
    assert result["execution"]["unique_request_count"] == 2
    assert result["execution"]["deduplicated_request_count"] == 2
    assert len(router.requests) == 2
