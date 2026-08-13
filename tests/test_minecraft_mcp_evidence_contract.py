from __future__ import annotations

from minecraft_mod_ai.central_research import normalize_research_brief
from minecraft_mod_ai.minecraft_mcp_evidence_contract import (
    collect_external_minecraft_evidence,
)


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


def test_minecraft_technical_domains_gain_external_mcp_route() -> None:
    brief = normalize_research_brief("Add a custom Fabric entity", {"title": "x"})
    request = next(
        domain for domain in brief["domains"] if domain["domain_id"] == "request"
    )
    assert "external_mcp" in request["providers"]


def test_external_evidence_is_batched_scoped_and_compact() -> None:
    brief = normalize_research_brief("Add a custom Fabric entity", {"title": "x"})
    router = _FakeRouter()
    result = collect_external_minecraft_evidence(brief, router=router)

    assert result["execution"]["parallel"] is True
    assert result["execution"]["single_flight_cache"] is True
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
