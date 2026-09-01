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
        "content": (
            "The implementation stores colony state, and colonies persist their population "
            "between server restarts."
        ),
    }
    accepted, rejected = quality._verify_page_claims(
        _FakeAgentic,
        _FakeBoundedRag,
        router=object(),
        domain_id="req_colony",
        page=page,
        claims=[
            "Colony population persists across restarts.",
            "Colonies automatically build rockets.",
        ],
        progress_label="test",
    )

    assert [item["claim"] for item in accepted] == [
        "Colony population persists across restarts."
    ]
    assert accepted[0]["support_quote"] == "colonies persist their population"
    assert accepted[0]["evidence_refs"] == ["sha256:doc#page=1/1"]
    assert rejected == ["Colonies automatically build rockets."]


def test_corrective_query_filter_uses_only_new_executable_queries():
    seen = {"minecraft colony progression"}
    result = quality._correction_queries(
        [
            "minecraft colony progression",
            "마인크래프트 식민지",
            "colony persistence saved state source",
            "colony persistence saved state source",
        ],
        seen=seen,
        raw_prompt="식민지를 만들어줘",
    )
    assert result == ["colony persistence saved state source"]


def test_corrective_loop_retrieves_next_query_before_fixed_point(monkeypatch, tmp_path):
    calls = {"forced": 0, "writes": []}

    def fake_read_and_verify(
        agentic_module,
        project_rag,
        router,
        *,
        prompt,
        domain,
        document,
        domain_key,
        failures,
        round_index,
    ):
        del agentic_module, project_rag, router, prompt, domain, domain_key, failures
        page = {
            "page_ref": f"{document['document_sha256']}#page=1/1",
            "content": "host evidence",
        }
        if round_index == 0:
            note = {
                "_host_page_ref": page["page_ref"],
                "domain_id": "req_colony",
                "claims": [],
                "gaps": ["Need persistent colony-state evidence."],
                "next_queries": ["minecraft colony persistent state source"],
                "procedures": [],
                "sufficient": False,
            }
        else:
            note = {
                "_host_page_ref": page["page_ref"],
                "domain_id": "req_colony",
                "claims": [
                    {
                        "claim": "Colony state can be persisted.",
                        "evidence_refs": [page["page_ref"]],
                        "support_quote": "host evidence",
                        "support_quote_sha256": "sha256:q",
                        "support_verification": "model_entailment+host_exact_quote",
                    }
                ],
                "gaps": [],
                "next_queries": [],
                "procedures": [],
                "sufficient": True,
            }
        return [page], [note], 0

    monkeypatch.setattr(corrective, "_read_and_verify_document", fake_read_and_verify)

    class FakeRag:
        _BoundedResearchOutputError = RuntimeError

        @staticmethod
        def _domain_checkpoint_key(router, *, prompt, domain, document):
            return "base"

        @staticmethod
        def _sha256(value):
            return "sha256:quality-key"

        @staticmethod
        def _domain_lock(key):
            return nullcontext()

        @staticmethod
        def _read_complete_manifest(agentic, key, domain_id):
            return None

        @staticmethod
        def _checkpoint_dir(key):
            return tmp_path

        @staticmethod
        def _manifest_path(key):
            return tmp_path / "manifest.json"

        @staticmethod
        def _prompt_document_receipt(document):
            return {
                "document_sha256": document["document_sha256"],
                "page_count": document.get("page_count", 1),
            }

        @staticmethod
        def _forced_rag_bundle(router, brief):
            calls["forced"] += 1
            assert brief["domains"][0]["queries"] == [
                "minecraft colony persistent state source"
            ]
            return {"domains": [{"domain_id": "req_colony", "queries": []}]}

        @staticmethod
        def _materialize_domain_evidence_document(domain_id, evidence):
            assert evidence["grounded_rag"]["queries"][0]["evidence_records"]
            return {"document_sha256": "corrected", "page_count": 1}

        @staticmethod
        def _materialize_claim_catalog(key, domain_id, claims):
            return {"claim_count": len(claims)}

        @staticmethod
        def _materialize_evidence_ledger(key, domain_id, pages):
            return {"record_count": len(pages)}

        @staticmethod
        def _write_manifest(key, *, status, note, failures):
            calls["writes"].append((status, note, failures))

    class FakePipeline:
        @staticmethod
        def _grounded_domain_evidence(agentic, domain_id, bundle):
            return {
                "domain_id": domain_id,
                "queries": [
                    {
                        "query": "minecraft colony persistent state source",
                        "query_sha256": "q",
                        "evidence_records": [
                            {
                                "source_id": "github:colony",
                                "source_type": "github_source",
                                "content": "Minecraft Fabric persistent colony saved state implementation",
                                "content_sha256": "sha256:new",
                            }
                        ],
                        "github_provider_status": "available",
                    }
                ],
            }

    class FakeAgentic:
        class SpecValidationError(ValueError):
            pass

        @staticmethod
        def _validate_sufficient_research(note, *, allowed_refs):
            assert note["claims"]
            assert all(
                ref in allowed_refs
                for claim in note["claims"]
                for ref in claim["evidence_refs"]
            )

    note = quality._quality_research_document_domain(
        FakePipeline,
        FakeAgentic,
        FakeRag,
        router=object(),
        prompt="식민지를 만들어줘",
        domain={
            "domain_id": "req_colony",
            "queries": ["minecraft colony progression"],
        },
        document={"document_sha256": "base-doc", "page_count": 1},
        trace_metadata=None,
    )

    assert calls["forced"] == 1
    assert note["sufficient"] is True
    assert note["quality_contract"]["corrective_rounds_executed"] == 1
    assert note["quality_contract"]["donor_selection_performed"] is False
    assert note["claims"][0]["claim"] == "Colony state can be persisted."