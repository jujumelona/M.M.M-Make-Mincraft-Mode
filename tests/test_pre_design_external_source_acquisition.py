from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import pre_design_external_source_contract as external
from minecraft_mod_ai import pre_design_rag_corrective as corrective
from minecraft_mod_ai import pre_design_research_pipeline as pipeline


def test_forced_rag_external_contract_materializes_real_source_body(monkeypatch) -> None:
    def base(_router, payload):
        query = payload["domains"][0]["queries"][0]
        return {
            "schema_version": "mmm/agentic-research-v1",
            "domains": [
                {
                    "domain_id": "request",
                    "queries": [
                        {
                            "query": query,
                            "query_sha256": "sha256:test",
                            "project_rag": {"records": []},
                        }
                    ],
                }
            ],
        }

    module = SimpleNamespace(_forced_rag_bundle=base)
    monkeypatch.setattr(
        external,
        "_retrieve_github_source_body",
        lambda query: {
            "records": [
                {
                    "source_id": "github:example/space-mod:abc",
                    "source_type": "github_source_body",
                    "source_locator": "https://github.com/example/space-mod/blob/main/README.md",
                    "url": "https://github.com/example/space-mod/blob/main/README.md",
                    "title": "example/space-mod",
                    "content": (
                        "Space exploration mod with planets, resource mining, trading, "
                        "spaceship upgrades and alien combat."
                    ),
                    "content_sha256": "sha256:body",
                    "body_retrieved": True,
                    "metadata": {"repository": "example/space-mod", "query": query},
                }
            ],
            "search_requests": 1,
            "source_requests": 1,
            "provider_status": "available",
            "saturation_reason": "source_body_retrieved",
            "errors": [],
        },
    )

    external.install(module)
    payload = {
        "schema_version": "mmm/corrective-retrieval-request-v1",
        "domains": [
            {
                "domain_id": "request",
                "requirements": ["space mod"],
                "providers": ["github"],
                "queries": ["minecraft mod space resource mining trading"],
            }
        ],
    }
    bundle = module._forced_rag_bundle(None, payload)
    row = bundle["domains"][0]["queries"][0]

    assert row["external_rag"]["github_retrieval"] == {
        "provider_status": "available",
        "saturation_reason": "source_body_retrieved",
        "search_requests": 1,
        "source_requests": 1,
    }
    records = pipeline._query_content_records(row)
    assert len(records) == 1
    assert records[0]["body_retrieved"] is True
    assert "resource mining" in records[0]["content"]

    receipt = pipeline._grounded_rag_receipt(bundle)
    query_receipt = receipt["domains"][0]["queries"][0]
    assert receipt["content_record_count"] == 1
    assert query_receipt["github_record_count"] == 1
    assert query_receipt["github_search_requests"] == 1
    assert query_receipt["github_source_requests"] == 1


def test_corrective_query_alias_search_queries_is_executable() -> None:
    class SpecValidationError(ValueError):
        pass

    agentic = SimpleNamespace(SpecValidationError=SpecValidationError)

    class ProjectRag:
        @staticmethod
        def _generate_bounded(_agentic, _router, **kwargs):
            return kwargs["parser"](
                '{"search_queries":["minecraft mod space economy trading currency"]}'
            )

    seen: set[str] = set()
    queries = corrective._generate_gap_queries(
        agentic,
        ProjectRag,
        None,
        domain={
            "domain_id": "request",
            "objective": "find source evidence",
            "requirements": ["space mod"],
        },
        gaps=["missing source evidence"],
        prior_queries=[],
        seen=seen,
        raw_prompt="우주 모드",
        progress_label="test",
    )

    assert queries == ["minecraft mod space economy trading currency"]


def test_external_metadata_without_body_never_becomes_evidence(monkeypatch) -> None:
    module = SimpleNamespace(
        _forced_rag_bundle=lambda _router, payload: {
            "domains": [
                {
                    "domain_id": "request",
                    "queries": [
                        {
                            "query": payload["domains"][0]["queries"][0],
                            "query_sha256": "sha256:test",
                        }
                    ],
                }
            ]
        }
    )
    monkeypatch.setattr(
        external,
        "_retrieve_github_source_body",
        lambda _query: {
            "records": [],
            "search_requests": 1,
            "source_requests": 3,
            "provider_status": "available",
            "saturation_reason": "repositories_found_no_claim_bearing_source_body",
            "errors": [],
        },
    )
    external.install(module)
    payload = {
        "schema_version": "mmm/corrective-retrieval-request-v1",
        "domains": [{"domain_id": "request", "queries": ["minecraft mod space colonies"]}],
    }
    bundle = module._forced_rag_bundle(None, payload)
    row = bundle["domains"][0]["queries"][0]

    assert pipeline._query_content_records(row) == []
    github = row["external_rag"]["github_retrieval"]
    assert github["search_requests"] == 1
    assert github["source_requests"] == 3
