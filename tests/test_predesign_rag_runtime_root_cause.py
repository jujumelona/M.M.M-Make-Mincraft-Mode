from __future__ import annotations

from pathlib import Path

import minecraft_mod_ai.agentic_pre_design_rag as rag
import minecraft_mod_ai.pre_design_external_source_contract as external


def _grounded(body: str) -> dict[str, object]:
    return {
        "grounded_rag": {
            "domain_id": "request",
            "fusion": {"diagnostic": "SHOULD_NOT_REACH_MODEL"},
            "queries": [
                {
                    "query": "minecraft mod travel space",
                    "github_provider_status": "available",
                    "github_saturation_reason": "source_body_retrieved",
                    "evidence_records": [
                        {
                            "source_id": "github:fixture/repo:abc",
                            "source_type": "github_source_body",
                            "source_locator": "https://example.invalid/repo",
                            "url": "https://example.invalid/repo",
                            "title": "fixture/repo",
                            "content": body,
                            "content_sha256": rag._sha256_text(body),
                            "metadata": {"repository": "fixture/repo", "query": "metadata-only"},
                        }
                    ],
                    "retrieval_errors": [],
                }
            ],
        }
    }


def test_model_evidence_projection_contains_source_body_not_retrieval_envelope(tmp_path, monkeypatch) -> None:
    body = "REAL SOURCE BODY describing space travel mechanics and ship progression. " * 50
    monkeypatch.setenv("MMM_WORKSPACE", str(tmp_path))
    document = rag._materialize_domain_evidence_document("request", _grounded(body))
    pages = rag._read_evidence_pages(document)
    joined = "\n".join(str(page["content"]) for page in pages)
    assert "REAL SOURCE BODY" in joined
    assert "SHOULD_NOT_REACH_MODEL" not in joined
    assert "github_provider_status" not in joined
    assert "github_saturation_reason" not in joined
    assert "retrieval_errors" not in joined
    assert document["model_projection"] == "claim_bearing_source_bodies_only;raw_receipt_lossless"


def test_unauthenticated_external_query_budget_stays_below_github_search_ceiling(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("MMM_PREDESIGN_EXTERNAL_SOURCE_QUERIES", "20")
    assert external._max_queries() == 8


def test_repository_search_query_is_bounded_and_not_full_natural_language() -> None:
    value = external._repository_search_query(
        "minecraft mod gather resources space environment with a very long tail"
    )
    assert value == "gather resources minecraft"
    assert "environment" not in value
    assert "very long tail" not in value
