from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import agentic_pre_design_rag as project_rag
from minecraft_mod_ai import pre_design_research_pipeline as subject


def _domain_value(domain_id, value):
    if not isinstance(value, dict):
        return {}
    for domain in value.get("domains", []):
        if isinstance(domain, dict) and domain.get("domain_id") == domain_id:
            return domain
    return {}


def test_target_neutral_official_rag_is_query_scoped_not_corpus_dump():
    evidence = subject._target_neutral_official_evidence(
        {
            "domains": [
                {
                    "domain_id": "resource",
                    "objective": "gather ore resources and harvesting progression",
                    "queries": ["resource gathering harvesting ore progression"],
                }
            ]
        }
    )
    documents = evidence["domains"][0]["documents"]
    ids = {str(item.get("document_id", "")) for item in documents}
    assert len(documents) < len(subject.BUILTIN_CORPUS)
    assert "fabric-blockbench" not in ids


def test_grounded_domain_evidence_drops_toc_metadata_and_keeps_source_content():
    bundle = {
        "domains": [
            {
                "domain_id": "resource",
                "queries": [
                    {
                        "query": "resource gathering harvesting",
                        "query_sha256": "sha256:q",
                        "project_rag": {
                            "sources": [
                                {
                                    "source_id": "fabric-blockbench",
                                    "title": "Blockbench Documentation",
                                    "url": "https://docs.fabricmc.net/develop/blockbench/",
                                }
                            ]
                        },
                        "external_rag": {
                            "status": "ok",
                            "documents": [
                                {
                                    "source_id": "github:owner/repo:src/Harvest.java",
                                    "url": "https://github.com/owner/repo/blob/main/src/Harvest.java",
                                    "content": "public final class Harvest { void gatherResourceNode() { collectOre(); } }",
                                    "metadata": {"repository": "owner/repo"},
                                }
                            ],
                        },
                    }
                ],
            }
        ]
    }
    agentic = SimpleNamespace(_domain_source_value=_domain_value)
    grounded = subject._grounded_domain_evidence(agentic, "resource", bundle)
    records = grounded["queries"][0]["evidence_records"]
    assert len(records) == 1
    assert records[0]["source_id"].startswith("github:")
    assert "Harvest" in records[0]["content"]
    assert all(record.get("title") != "Blockbench Documentation" for record in records)


def test_required_github_gap_fails_before_model_extraction():
    with pytest.raises(subject.PreDesignResearchFailure, match="github"):
        subject._validate_domain_provider_grounding(
            {"domain_id": "resource", "providers": ["github"]},
            {
                "queries": [
                    {
                        "query": "resource gathering",
                        "evidence_records": [
                            {
                                "source_id": "official:resource-doc",
                                "content": "A sufficiently long official resource gathering explanation for the test.",
                            }
                        ],
                    }
                ]
            },
        )


def test_live_collect_design_research_uses_unified_grounded_owner(monkeypatch):
    calls = []
    brief = {
        "summary": "resource research",
        "domains": [
            {
                "domain_id": "resource",
                "objective": "resource gathering",
                "queries": ["resource gathering"],
                "providers": ["github"],
            }
        ],
        "unresolved_questions": [],
    }
    plan = {
        "policy": {"target_frozen": False},
        "research_domains": [],
        "plan_sha256": "sha256:plan",
    }
    bundle = {
        "schema_version": "mmm/forced-pre-design-rag",
        "research_sha256": "sha256:rag",
        "domains": [
            {
                "domain_id": "resource",
                "queries": [
                    {
                        "query": "resource gathering",
                        "query_sha256": "sha256:q",
                        "external_rag": {
                            "status": "ok",
                            "documents": [
                                {
                                    "source_id": "github:owner/repo:src/Harvest.java",
                                    "url": "https://github.com/owner/repo/blob/main/src/Harvest.java",
                                    "content": "public final class Harvest { void gatherResourceNode() { collectOre(); } }",
                                    "metadata": {"repository": "owner/repo"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    monkeypatch.setattr(subject, "_pre_design_brief", lambda prompt: brief)
    monkeypatch.setattr(subject, "compile_minecraft_knowledge_plan", lambda prompt: plan)

    def forced(router, research_brief):
        calls.append((router, research_brief))
        return bundle

    monkeypatch.setattr(project_rag, "_forced_rag_bundle", forced)
    monkeypatch.setattr(
        project_rag,
        "_materialize_domain_evidence_document",
        lambda domain_id, evidence: {"domain_id": domain_id, "evidence": evidence},
    )
    monkeypatch.setattr(subject, "_validate_document_grounding", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        subject,
        "research_document_domain",
        lambda *args, **kwargs: {
            "domain_id": "resource",
            "research_failures": [],
            "sufficient": True,
            "fixed_point": True,
            "checkpoint": {"status": "complete"},
        },
    )
    monkeypatch.setattr(
        subject,
        "evaluate_route_coverage",
        lambda *args, **kwargs: {"status": "PASS", "blocking_requirement_refs": []},
    )
    monkeypatch.setattr(
        subject, "attach_procedural_skillbank", lambda router, prompt, payload: payload
    )
    monkeypatch.setattr(
        subject, "compose_research_skillbank", lambda router, prompt, payload: payload
    )
    monkeypatch.setattr(
        subject,
        "_bounded_model_view",
        lambda agentic, router, prompt, payload: {
            "model_view_sha256": "sha256:view",
            "payload": payload,
        },
    )

    result = subject.collect_design_research(object(), "resource gathering")
    assert len(calls) == 1
    deterministic = result["payload"]["deterministic"]
    assert "grounded_rag" in deterministic
    assert "official_rag" not in deterministic
    assert "forced_project_rag" not in deterministic
