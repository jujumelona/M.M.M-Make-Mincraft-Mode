from __future__ import annotations

import urllib.error
from pathlib import Path

from minecraft_mod_ai import pre_design_grounded_rag as rag
from minecraft_mod_ai import small_model_predesign_research as small
from minecraft_mod_ai.pre_design_rag_quality_contract import _source_body


def test_github_403_does_not_stop_remaining_requirements(monkeypatch):
    queries = [f"requirement {index} unique mechanic" for index in range(11)]
    brief = {"domains": [{"domain_id": "request", "queries": queries}]}
    modrinth_calls: list[str] = []
    github_calls: list[str] = []

    def modrinth(query: str):
        modrinth_calls.append(query)
        body = f"verified implementation evidence for {query}"
        return [
            {
                "source_id": f"modrinth:{len(modrinth_calls)}",
                "source_type": "modrinth_project_body",
                "source_locator": f"modrinth:{len(modrinth_calls)}",
                "url": "https://modrinth.com/mod/example",
                "title": "example",
                "content": body,
                "content_sha256": rag._sha256_text(body),
                "body_retrieved": True,
            }
        ], {"provider": "modrinth", "status": "available", "result_count": 1}

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
    monkeypatch.setattr(
        rag,
        "_search_curseforge",
        lambda query: ([], {"provider": "curseforge", "status": "not_configured", "result_count": 0}),
    )
    monkeypatch.setattr(rag, "_search_github", github)
    monkeypatch.setattr(
        rag,
        "_search_authoritative_catalog",
        lambda query, versions: {"sources": [], "errors": []},
    )
    monkeypatch.setattr(rag, "_existing_code_index", lambda: None)
    monkeypatch.setattr(
        rag,
        "_search_code_index",
        lambda index, query: {"status": "not_indexed", "hits": []},
    )

    bundle = rag._forced_rag_bundle(object(), brief)

    assert bundle["query_count"] == 11
    assert len(modrinth_calls) == 11
    assert len(github_calls) == 11
    rows = bundle["domains"][0]["queries"]
    assert len(rows) == 11
    assert all(row["external_rag"]["sources"] for row in rows)
    assert rows[0]["external_rag"]["github_retrieval"]["provider_status"] == "error"
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


def test_zero_source_bodies_skip_model_and_preserve_requirement(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    evidence = {
        "grounded_rag": {
            "domain_id": "request",
            "queries": [
                {
                    "query": "authored mechanic",
                    "evidence_records": [],
                    "content_record_count": 0,
                }
            ],
        }
    }
    document = rag._materialize_domain_evidence_document("request", evidence)
    assert document["model_unit_count"] == 0
    assert document["page_count"] == 0

    class ExplodingRouter:
        calls = 0

        def generate_text(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("model must not be called when source body count is zero")

    router = ExplodingRouter()
    note = small.research_document_domain(
        object(),
        rag,
        router,
        prompt="build the authored mechanic",
        domain={
            "domain_id": "request",
            "objective": "build the authored mechanic",
            "requirements": ["the player can activate the authored mechanic"],
            "queries": ["authored mechanic"],
        },
        document=document,
        trace_metadata=None,
    )

    assert router.calls == 0
    assert note["model_called"] is False
    assert note["source_body_count"] == 0
    assert note["authoritative_requirement_fallback"] == [
        "the player can activate the authored mechanic"
    ]
    assert "model_not_called" in note["page_local_diagnostics"][0]


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
