from __future__ import annotations

from contextlib import nullcontext

from minecraft_mod_ai import pre_design_rag_corrective as corrective
from minecraft_mod_ai import pre_design_rag_quality_contract as quality


def test_fusion_deduplicates_across_queries_and_rewards_multi_query_support():
    shared_content = "Minecraft colony progression settlement population buildings"
    grounded = {
        "domain_id": "req_colony",
        "queries": [
            {
                "query": "minecraft colony progression",
                "query_sha256": "q1",
                "evidence_records": [
                    {
                        "source_id": "github:first",
                        "content": shared_content,
                        "content_sha256": "sha256:shared",
                        "source_type": "github_source",
                    },
                    {
                        "source_id": "modrinth:noise",
                        "content": "Generic unrelated weather metadata and release notes.",
                        "content_sha256": "sha256:noise",
                        "source_type": "modrinth_project",
                    },
                ],
                "github_provider_status": "available",
            },
            {
                "query": "colony settlement mod source",
                "query_sha256": "q2",
                "evidence_records": [
                    {
                        "source_id": "github:duplicate",
                        "content": shared_content,
                        "content_sha256": "sha256:shared",
                        "source_type": "github_source",
                    }
                ],
                "github_provider_status": "available",
            },
        ],
    }

    fused = quality.fuse_grounded_domain_evidence({}, grounded)
    records = fused["queries"][0]["evidence_records"]

    assert len(records) == 1
    assert fused["fusion"]["duplicate_record_count"] == 1
    assert fused["fusion"]["zero_relevance_dropped_record_count"] == 1
    assert fused["retrieval_trace"][0]["zero_relevance_dropped"] == 1
    assert fused["fusion"]["query_coverage_ratio"] == 1.0
    shared = records[0]
    assert shared["content_sha256"] == "sha256:shared"
    assert shared["retrieval_fusion"]["query_hits"] == 2
    assert shared["retrieval_fusion"]["matched_queries"] == [
        "minecraft colony progression",
        "colony settlement mod source",
    ]


def test_retrieval_query_filter_rejects_raw_prompt_and_non_ascii():
    prompt = "make a colony progression mod"
    assert quality._is_retrieval_query(
        "minecraft colony progression source",
        raw_prompt=prompt,
    )
    assert not quality._is_retrieval_query(prompt, raw_prompt=prompt)
    assert not quality._is_retrieval_query(
        "마인크래프트 colony progression",
        raw_prompt=prompt,
    )


class _SpecValidationError(ValueError):
    pass


class _FakeAgentic:
    SpecValidationError = _SpecValidationError


class _FakeBoundedRag:
    @staticmethod
    def _generate_bounded(
        agentic_module,
        router,
        *,
        messages,
        response_schema,
        parser,
        progress_label,
    ):
        del agentic_module, router, messages, response_schema, progress_label
        return parser(
            '{"verdicts":['
            '{"claim_index":0,"supported":true,'
            '"support_quote":"colonies persist their population"},'
            '{"claim_index":1,"supported":false,"support_quote":""}'
            "]}"
        )


def test_claim_support_requires_exact_host_quote():
    page = {
        "page_ref": "sha256:doc#page=1/1",
        "content": "The source says colonies persist their population across saves.",
        "content_sha256": "sha256:doc",
    }
    claims = [
        {"claim": "colonies persist", "evidence_refs": [page["page_ref"]]},
        {"claim": "weather exists", "evidence_refs": [page["page_ref"]]},
    ]
    supported, unresolved = corrective._verify_claim_support(
        _FakeAgentic,
        _FakeBoundedRag,
        router=object(),
        domain_id="req_colony",
        claims=claims,
        pages=[page],
    )

    assert supported[0]["supported"] is True
    assert supported[0]["support_quote"] == "colonies persist their population"
    assert supported[1]["supported"] is False
    assert unresolved == ["weather exists"]


def test_minecraft_route_requires_verified_evidence_not_metadata_only():
    record = {
        "source_id": "github:example/repo",
        "url": "https://github.com/example/repo",
        "content": "A README about unrelated tooling.",
        "content_sha256": "sha256:body",
        "source_type": "github_source",
    }
    route = corrective._minecraft_knowledge_route_receipt(
        domain={"id": "req_colony", "summary": "colony settlements"},
        evidence_records=[record],
        verified_records=[],
    )
    assert route["status"] == "BLOCKED"
    assert route["verified_evidence_count"] == 0


def test_corrective_pipeline_keeps_page_local_gaps_out_of_domain_blockers(monkeypatch):
    domain = {
        "id": "req_colony",
        "summary": "persistent colony population and buildings",
        "research_questions": ["minecraft colony persistence source"],
    }
    evidence = {
        "queries": [
            {
                "query": "domain fused evidence",
                "source_queries": ["minecraft colony persistence source"],
                "evidence_records": [
                    {
                        "source_id": "github:example/colony",
                        "url": "https://github.com/example/colony",
                        "content": "Colonies persist their population and buildings across saves.",
                        "content_sha256": "sha256:colony",
                        "source_type": "github_source",
                    }
                ],
            }
        ],
        "fusion": {"query_coverage_ratio": 1.0},
    }

    monkeypatch.setattr(
        corrective,
        "_materialize_evidence_pages",
        lambda *args, **kwargs: (
            [
                {
                    "page_ref": "sha256:colony#page=1/1",
                    "content": "Colonies persist their population and buildings across saves.",
                    "content_sha256": "sha256:colony",
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        corrective,
        "_generate_claims",
        lambda *args, **kwargs: [
            {
                "claim": "colonies persist population",
                "evidence_refs": ["sha256:colony#page=1/1"],
            },
            {
                "claim": "unverified economy behavior",
                "evidence_refs": ["sha256:colony#page=1/1"],
            },
        ],
    )
    monkeypatch.setattr(
        corrective,
        "_verify_claim_support",
        lambda *args, **kwargs: (
            [
                {
                    "claim": "colonies persist population",
                    "supported": True,
                    "support_quote": "Colonies persist their population",
                    "evidence_refs": ["sha256:colony#page=1/1"],
                },
                {
                    "claim": "unverified economy behavior",
                    "supported": False,
                    "support_quote": "",
                    "evidence_refs": ["sha256:colony#page=1/1"],
                },
            ],
            ["unverified economy behavior"],
        ),
    )
    monkeypatch.setattr(
        corrective,
        "_minecraft_knowledge_route_receipt",
        lambda **kwargs: {
            "status": "PASS",
            "verified_evidence_count": 1,
            "target_frozen": False,
        },
    )

    result = corrective.run_corrective_research(
        _FakeAgentic,
        _FakeBoundedRag,
        router=object(),
        domain=domain,
        evidence=evidence,
        minecraft_target={"version": "1.21.8"},
    )

    assert result["complete"] is True
    assert result["domain_blocking_gaps"] == []
    assert result["unresolved_gaps"] == ["unverified economy behavior"]
