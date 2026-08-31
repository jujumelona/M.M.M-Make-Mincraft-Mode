from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import minecraft_mod_ai.agentic_pre_design_rag as rag
import minecraft_mod_ai.pre_design_external_source_contract as external
import minecraft_mod_ai.pre_design_rag_corrective as corrective


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


def test_one_recoverable_page_failure_does_not_poison_verified_domain(monkeypatch, tmp_path) -> None:
    page_ref = "sha256:page#page=1/1"

    def fake_read(*args, failures, **kwargs):
        failures.append({"unit": "page:0:0", "error": "fixture parse failure on sibling fragment"})
        pages = [{"page_ref": page_ref, "content": "verified body"}]
        notes = [
            {
                "_host_page_ref": page_ref,
                "domain_id": "request",
                "claims": [
                    {
                        "claim": "Verified source-backed mechanic exists.",
                        "evidence_refs": [page_ref],
                        "support_quote": "verified body",
                        "support_quote_sha256": "sha256:quote",
                        "support_verification": "model_entailment+host_exact_quote",
                    }
                ],
                "gaps": [],
                "next_queries": [],
                "procedures": [],
                "sufficient": True,
            }
        ]
        return pages, notes, 0

    monkeypatch.setattr(corrective, "_read_and_verify_document", fake_read)
    written: dict[str, object] = {}
    project_rag = SimpleNamespace(
        _domain_checkpoint_key=lambda *a, **k: "base",
        _sha256=lambda value: "sha256:domain-key",
        _domain_lock=lambda key: nullcontext(),
        _read_complete_manifest=lambda *a, **k: None,
        _materialize_claim_catalog=lambda *a, **k: {"claim_count": len(a[-1])},
        _materialize_evidence_ledger=lambda *a, **k: {"page_count": len(a[-1])},
        _prompt_document_receipt=lambda document: {"document_sha256": document.get("document_sha256")},
        _manifest_path=lambda key: tmp_path / "manifest.json",
        _checkpoint_dir=lambda key: tmp_path,
        _write_manifest=lambda key, **kwargs: written.update(kwargs),
    )
    agentic = SimpleNamespace(_validate_sufficient_research=lambda *a, **k: None)
    note = corrective._quality_research_document_domain(
        SimpleNamespace(),
        agentic,
        project_rag,
        SimpleNamespace(),
        prompt="space mod",
        domain={"domain_id": "request", "queries": ["space mod mechanics"]},
        document={"document_sha256": "sha256:document"},
        trace_metadata=None,
    )
    assert note["sufficient"] is True
    assert note["fixed_point"] is False
    assert note["quality_contract"]["fixed_point_reason"] == "verified_claims_sufficient"
    assert "research_failures" not in note
    assert note["research_warnings"]
    assert written["status"] == "complete"
