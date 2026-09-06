from __future__ import annotations

import urllib.error
from pathlib import Path

from minecraft_mod_ai import catalog_first_grounded_rag as catalog_rag
from minecraft_mod_ai import pre_design_grounded_rag as rag
from minecraft_mod_ai.pre_design_rag_quality_contract import _source_body


def test_github_403_does_not_stop_remaining_requirements(monkeypatch):
    queries = [f"requirement {index} unique mechanic" for index in range(11)]
    brief = {
        "domains": [
            {
                "domain_id": "request",
                "providers": ["modrinth"],
                "queries": queries,
            }
        ]
    }
    modrinth_calls: list[str] = []
    github_calls: list[str] = []

    def modrinth(query: str):
        modrinth_calls.append(query)
        return [], {"provider": "modrinth", "status": "available", "result_count": 0}

    def github(query: str, *, disabled=None, disable=None):
        github_calls.append(query)
        if disabled is not None and disabled():
            return [], {
                "provider": "github",
                "status": "disabled_after_rate_or_auth_failure",
                "result_count": 0,
            }
        if disable is not None:
            disable()
        raise urllib.error.HTTPError(
            "https://api.github.com/search/repositories", 403, "rate limited", {}, None
        )

    monkeypatch.setattr(rag, "_search_modrinth", modrinth)
    monkeypatch.setattr(rag, "_search_github", github)

    bundle = catalog_rag.forced_rag_bundle(rag, object(), brief)

    assert bundle["query_count"] == 11
    assert len(modrinth_calls) == 11
    assert len(github_calls) == 11
    rows = bundle["domains"][0]["queries"]
    assert len(rows) == 11
    assert all(not row["external_rag"]["sources"] for row in rows)
    assert all(
        row["external_rag"]["github_retrieval"]["provider_status"]
        in {"error", "disabled_after_rate_or_auth_failure"}
        for row in rows
    )


def test_fetched_modrinth_body_is_claim_bearing_evidence():
    body = "A concrete implementation body with persistence and interaction details."
    record = {
        "source_id": "modrinth:abc123",
        "source_type": "modrinth_project_body",
        "source_locator": "modrinth:abc123",
        "url": "https://modrinth.com/mod/example",
        "content": body,
        "body_retrieved": True,
    }
    assert _source_body(record) == body


def test_materializer_never_creates_fake_empty_page(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    document = rag._materialize_domain_evidence_document(
        "request", {"grounded_rag": {"domain_id": "request", "queries": []}}
    )
    assert document["model_unit_count"] == 0
    assert document["page_count"] == 0
    assert rag._read_evidence_pages(document) == []


def test_legacy_pre_design_owner_is_physically_absent_and_unreferenced():
    root = Path("minecraft_mod_ai")
    legacy = root / "agentic_pre_design_rag.py"
    assert not legacy.exists()
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "agentic_pre_design_rag" in text:
            offenders.append(str(path))
    assert offenders == []


def test_runtime_does_not_reinstall_retired_predesign_wrappers():
    bootstrap = Path("minecraft_mod_ai/runtime_bootstrap.py").read_text(encoding="utf-8")
    finalization = Path("minecraft_mod_ai/runtime_finalization.py").read_text(encoding="utf-8")
    stability = Path("minecraft_mod_ai/runtime_stability_contract.py").read_text(encoding="utf-8")
    retrieval = Path("minecraft_mod_ai/small_model_retrieval_efficiency_contract.py").read_text(encoding="utf-8")

    assert "pre_design_external_source_contract" not in bootstrap
    assert "research_grounded_rag_contract" not in finalization
    assert "_install_bounded_research_efficiency" not in stability
    assert "_install_synthesis_convergence" not in stability
    assert "_install_pre_design_rag_cascade(agentic_pre_design_rag)" not in retrieval


def test_curseforge_is_a_valid_central_research_provider():
    from minecraft_mod_ai import central_research as central

    domain = central._research_domain(
        {
            "domain_id": "request",
            "objective": "research request",
            "requirements": ["request"],
            "evidence_kinds": ["runtime_behavior"],
            "queries": ["minecraft mechanic implementation"],
            "providers": ["modrinth", "curseforge", "github"],
        }
    )
    assert "curseforge" in domain.providers


def test_duplicate_queries_are_executed_once(monkeypatch):
    calls: list[str] = []

    def modrinth(query: str):
        calls.append(query)
        return [], {"provider": "modrinth", "status": "available", "result_count": 0}

    monkeypatch.setattr(rag, "_search_modrinth", modrinth)
    monkeypatch.setattr(
        rag,
        "_search_github",
        lambda query, **kwargs: (
            [],
            {"provider": "github", "status": "available", "result_count": 0},
        ),
    )
    brief = {
        "domains": [
            {
                "domain_id": "request",
                "providers": ["modrinth"],
                "queries": ["same query", "same query", "other query"],
            }
        ]
    }
    bundle = catalog_rag.forced_rag_bundle(rag, object(), brief)
    assert bundle["query_count"] == 3
    assert bundle["unique_query_count"] == 2
    assert sorted(calls) == ["other query", "same query"]


def test_catalog_candidate_without_source_link_never_triggers_broad_github_search(monkeypatch):
    body = "implementation body"
    monkeypatch.setattr(
        rag,
        "_search_modrinth",
        lambda query: (
            [
                {
                    "source_id": "modrinth:one",
                    "source_type": "modrinth_project_body",
                    "source_locator": "modrinth:one",
                    "url": "https://modrinth.com/mod/one",
                    "title": "one",
                    "content": body,
                    "content_sha256": rag._sha256_text(body),
                    "body_retrieved": True,
                    "metadata": {"source_url": ""},
                }
            ],
            {"provider": "modrinth", "status": "available", "result_count": 1},
        ),
    )

    def broad_search_must_not_run(*args, **kwargs):
        raise AssertionError(f"broad GitHub search escaped catalog-first policy: {args!r} {kwargs!r}")

    monkeypatch.setattr(rag, "_search_github", broad_search_must_not_run)
    bundle = catalog_rag.forced_rag_bundle(
        rag,
        object(),
        {
            "domains": [
                {
                    "domain_id": "request",
                    "providers": ["modrinth"],
                    "queries": ["one query"],
                }
            ]
        },
    )
    row = bundle["domains"][0]["queries"][0]
    source_ids = {
        source["source_id"] for source in row["external_rag"]["sources"]
    }
    assert source_ids == {"modrinth:one"}
    assert row["external_rag"]["github_retrieval"]["provider_status"] == (
        "skipped_catalog_without_linked_source"
    )
    assert row["external_rag"]["provider_policy"]["github_broad_search"] == (
        "fallback_only_after_empty_catalog"
    )


def test_query_terms_do_not_drop_late_authored_terms():
    query = "alpha beta gamma delta epsilon zeta theta iota kappa lambda nebula nova nexus orchid"
    terms = rag._query_terms(query).split()
    assert terms[-3:] == ["nova", "nexus", "orchid"]
    assert len(terms) == 14


def test_authoritative_catalog_returns_all_applicable_primary_sources():
    result = rag._search_authoritative_catalog("automated testing data generation", ())
    source_ids = {source["source_id"] for source in result["sources"]}
    assert source_ids == {
        "fabric-project-creation",
        "fabric-building",
        "fabric-data-generation",
        "fabric-automatic-testing",
        "fabric-mod-json",
        "fabric-develop-live",
        "fabric-meta",
        "fabric-api-maven",
    }


def test_linked_github_source_discovery_does_not_stop_after_two(monkeypatch):
    records = [
        {"metadata": {"source_url": f"https://github.com/example/repo-{index}"}}
        for index in range(4)
    ]
    monkeypatch.setattr(rag, "_text", lambda url, headers=None: "implementation body")
    found, receipt = rag._linked_github_sources(
        records, disabled=lambda: False, disable=lambda: None
    )
    assert len(found) == 4
    assert receipt["source_requests"] == 4
