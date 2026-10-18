from __future__ import annotations

from contextlib import nullcontext

from minecraft_mod_ai import pre_design_rag_corrective as corrective
from minecraft_mod_ai import pre_design_rag_fusion as fusion
from minecraft_mod_ai import pre_design_research_pipeline as pipeline
from minecraft_mod_ai import research_grounded_rag_contract as grounded


def test_removed_external_source_transports_have_no_executable_entrypoint():
    for name in ("_source_document", "_external_retrieval", "_modrinth_search",
                 "_github_adaptive_search"):
        assert not hasattr(grounded, name)


def test_pipeline_rejects_snippet_before_materialization():
    query_item = {
        "query": "colony persistent state",
        "external_rag": {
            "documents": [
                {
                    "source_id": "github:example/repo:search-result",
                    "source_type": "github_search_result",
                    "snippet": "colony persistent state implementation",
                }
            ]
        },
    }

    assert pipeline._query_content_records(query_item) == []


def test_grounded_receipt_status_tracks_real_body_not_candidate_metadata():
    metadata_bundle = {
        "domains": [
            {
                "domain_id": "request",
                "queries": [
                    {
                        "query": "colony persistent state",
                        "external_rag": {
                            "documents": [
                                {
                                    "source_id": "github:example/repo:search-result",
                                    "source_type": "github_search_result",
                                    "snippet": "colony persistent state implementation",
                                }
                            ]
                        },
                    }
                ],
            }
        ]
    }
    body_bundle = {
        "domains": [
            {
                "domain_id": "request",
                "queries": [
                    {
                        "query": "colony persistent state",
                        "external_rag": {
                            "documents": [
                                {
                                    "source_id": "github:example/repo:State.java",
                                    "source_type": "github_source",
                                    "content": "colony persistent state implementation",
                                }
                            ]
                        },
                    }
                ],
            }
        ]
    }

    metadata_receipt = pipeline._grounded_rag_receipt(metadata_bundle)
    body_receipt = pipeline._grounded_rag_receipt(body_bundle)

    assert metadata_receipt["status"] == "no_external_or_local_source_bodies"
    assert metadata_receipt["content_record_count"] == 0
    assert body_receipt["status"] == "available"
    assert body_receipt["content_record_count"] == 1


def test_fusion_projection_keeps_source_hash_and_hashes_model_facing_content(
    monkeypatch,
):
    monkeypatch.setenv("MMM_PREDESIGN_EVIDENCE_EXCERPT_CHARS", "800")
    body = "colony " + ("x" * 4_000) + " persistent state storage"
    source_digest = fusion._sha256_text(body)
    grounded_evidence = {
        "queries": [
            {
                "query": "colony persistent state",
                "evidence_records": [
                    {
                        "source_id": "github:example/repo:State.java",
                        "source_type": "github_source",
                        "content": body,
                        "content_sha256": source_digest,
                    }
                ],
            }
        ]
    }

    result = fusion.fuse_grounded_domain_evidence({}, grounded_evidence)
    record = result["queries"][0]["evidence_records"][0]

    assert record["content"] != body
    assert record["source_content_sha256"] == source_digest
    assert record["content_sha256"] == fusion._sha256_text(record["content"])
    assert record["retrieval_fusion"]["original_content_chars"] == len(body)
    assert record["retrieval_fusion"]["selected_content_chars"] < len(body)
    assert all(field not in record for field in ("body", "text", "snippet", "excerpt"))


def test_production_verified_claim_survives_page_local_gap(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("MMM_PREDESIGN_CORRECTIVE_ROUNDS", "0")
    writes: list[tuple[str, dict]] = []
    validations: list[tuple[dict, frozenset[str]]] = []
    page = {
        "page_ref": "sha256:base#page=1/1",
        "content": "colony state persistence partial evidence",
    }
    claim = {
        "claim": "Colony state can be persisted.",
        "evidence_refs": [page["page_ref"]],
        "support_quote": "colony state persistence",
        "support_verification": "model_entailment+host_exact_quote",
    }

    monkeypatch.setattr(
        corrective,
        "_read_and_verify_document",
        lambda *args, **kwargs: (
            [page],
            [
                {
                    "_host_page_ref": page["page_ref"],
                    "domain_id": "req_colony",
                    "claims": [claim],
                    "gaps": ["Need restart recovery evidence."],
                    "next_queries": ["minecraft colony restart recovery source"],
                    "procedures": [],
                    "sufficient": True,
                }
            ],
            0,
        ),
    )

    class FakeRag:
        _BoundedResearchOutputError = RuntimeError

        @staticmethod
        def _domain_checkpoint_key(router, *, prompt, domain, document):
            del router, prompt, domain, document
            return "base"

        @staticmethod
        def _sha256(value):
            del value
            return "sha256:worker02-page-local-gap"

        @staticmethod
        def _domain_lock(key):
            del key
            return nullcontext()

        @staticmethod
        def _read_complete_manifest(agentic, key, domain_id):
            del agentic, key, domain_id
            return None

        @staticmethod
        def _materialize_claim_catalog(key, domain_id, claims):
            del key, domain_id
            return {"claim_count": len(claims)}

        @staticmethod
        def _materialize_evidence_ledger(key, domain_id, pages):
            del key, domain_id
            return {"record_count": len(pages)}

        @staticmethod
        def _prompt_document_receipt(document):
            return {"document_sha256": document.get("document_sha256")}

        @staticmethod
        def _manifest_path(key):
            del key
            return tmp_path / "manifest.json"

        @staticmethod
        def _checkpoint_dir(key):
            del key
            return tmp_path

        @staticmethod
        def _write_manifest(key, *, status, note, failures):
            del key, failures
            writes.append((status, dict(note)))

    class FakeAgentic:
        class SpecValidationError(ValueError):
            pass

        @staticmethod
        def _validate_sufficient_research(note, *, allowed_refs):
            validations.append((dict(note), frozenset(allowed_refs)))

    result = corrective._quality_research_document_domain(
        object(),
        FakeAgentic,
        FakeRag,
        router=object(),
        prompt="Build a persistent colony system.",
        domain={
            "domain_id": "req_colony",
            "queries": ["minecraft colony persistent state source"],
        },
        document={"document_sha256": "base-doc"},
        trace_metadata=None,
    )

    assert writes
    status, note = writes[-1]
    assert status == "complete"
    assert result["sufficient"] is True
    assert result["fixed_point"] is False
    assert result["gaps"] == []
    assert result["page_local_gaps"] == ["Need restart recovery evidence."]
    assert result["quality_contract"]["fixed_point_reason"] == (
        "verified_claims_sufficient"
    )
    assert result["quality_contract"]["page_local_gap_count"] == 1
    assert validations
    assert validations[-1][1] == frozenset({page["page_ref"]})
    assert note["sufficient"] is True
