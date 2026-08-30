from __future__ import annotations

from contextlib import nullcontext

from minecraft_mod_ai import pre_design_rag_quality_contract as quality


def test_fusion_deduplicates_across_queries_and_rewards_multi_query_support():
    content = "Minecraft colony progression settlement population buildings"
    grounded = {
        "queries": [
            {
                "query": "minecraft colony progression",
                "evidence_records": [
                    {
                        "source_id": "github:first",
                        "content": content,
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
                "evidence_records": [
                    {
                        "source_id": "github:duplicate",
                        "content": content,
                        "content_sha256": "sha256:shared",
                        "source_type": "github_source",
                    }
                ],
                "github_provider_status": "available",
            },
        ]
    }
    fused = quality.fuse_grounded_domain_evidence({}, grounded)
    records = fused["queries"][0]["evidence_records"]
    assert len(records) == 2
    assert fused["fusion"]["duplicate_record_count"] == 1
    shared = next(item for item in records if item["content_sha256"] == "sha256:shared")
    assert shared["retrieval_fusion"]["query_hits"] == 2
    assert records[0]["content_sha256"] == "sha256:shared"


def test_query_filter_rejects_raw_prompt_and_non_ascii():
    prompt = "make a colony progression mod"
    assert quality._is_retrieval_query("minecraft colony progression source", raw_prompt=prompt)
    assert not quality._is_retrieval_query(prompt, raw_prompt=prompt)
    assert not quality._is_retrieval_query("마인크래프트 colony progression", raw_prompt=prompt)


class _SpecValidationError(ValueError):
    pass


class _Agentic:
    SpecValidationError = _SpecValidationError


class _Bounded:
    @staticmethod
    def _generate_bounded(agentic_module, router, *, messages, response_schema, parser, progress_label):
        del agentic_module, router, messages, response_schema, progress_label
        return parser(
            '{"verdicts":['
            '{"claim_index":0,"supported":true,"support_quote":"colonies persist their population"},'
            '{"claim_index":1,"supported":false,"support_quote":""}'
            ']}'
        )


def test_claim_support_requires_exact_host_quote():
    accepted, rejected = quality._verify_page_claims(
        _Agentic,
        _Bounded,
        router=object(),
        domain_id="req_colony",
        page={
            "page_ref": "sha256:doc#page=1/1",
            "content": "The implementation stores colony state, and colonies persist their population between server restarts.",
        },
        claims=[
            "Colony population persists across restarts.",
            "Colonies automatically build rockets.",
        ],
        progress_label="test",
    )
    assert [item["claim"] for item in accepted] == ["Colony population persists across restarts."]
    assert accepted[0]["support_quote"] == "colonies persist their population"
    assert rejected == ["Colonies automatically build rockets."]


def test_corrective_loop_retrieves_next_query_before_fixed_point(monkeypatch, tmp_path):
    calls = {"forced": 0}

    def fake_read(agentic_module, project_rag, router, *, prompt, domain, document, domain_key, failures, round_index):
        del agentic_module, project_rag, router, prompt, domain, domain_key, failures
        page = {"page_ref": f"{document['document_sha256']}#page=1/1", "content": "host evidence"}
        if round_index == 0:
            note = {
                "_host_page_ref": page["page_ref"],
                "claims": [],
                "gaps": ["Need persistent colony-state evidence."],
                "next_queries": ["minecraft colony persistent state source"],
                "procedures": [],
            }
        else:
            note = {
                "_host_page_ref": page["page_ref"],
                "claims": [{
                    "claim": "Colony state can be persisted.",
                    "evidence_refs": [page["page_ref"]],
                    "support_quote": "host evidence",
                    "support_quote_sha256": "sha256:q",
                    "support_verification": "model_entailment+host_exact_quote",
                }],
                "gaps": [],
                "next_queries": [],
                "procedures": [],
            }
        return [page], [note], 0

    monkeypatch.setattr(quality, "_read_and_verify_document", fake_read)

    class Rag:
        _BoundedResearchOutputError = RuntimeError
        _domain_checkpoint_key = staticmethod(lambda router, **kwargs: "base")
        _sha256 = staticmethod(lambda value: "sha256:quality-key")
        _domain_lock = staticmethod(lambda key: nullcontext())
        _read_complete_manifest = staticmethod(lambda agentic, key, domain_id: None)
        _checkpoint_dir = staticmethod(lambda key: tmp_path)
        _manifest_path = staticmethod(lambda key: tmp_path / "manifest.json")
        _prompt_document_receipt = staticmethod(lambda doc: {"document_sha256": doc["document_sha256"], "page_count": doc.get("page_count", 1)})
        _materialize_claim_catalog = staticmethod(lambda key, domain_id, claims: {"claim_count": len(claims)})
        _materialize_evidence_ledger = staticmethod(lambda key, domain_id, pages: {"record_count": len(pages)})
        _write_manifest = staticmethod(lambda *args, **kwargs: None)
        _emit_research_progress = staticmethod(lambda *args, **kwargs: None)

        @staticmethod
        def _forced_rag_bundle(router, brief):
            calls["forced"] += 1
            assert brief["domains"][0]["queries"] == ["minecraft colony persistent state source"]
            return {"domains": [{"domain_id": "req_colony", "queries": []}]}

        @staticmethod
        def _materialize_domain_evidence_document(domain_id, evidence):
            assert evidence["grounded_rag"]["queries"][0]["evidence_records"]
            return {"document_sha256": "corrected", "page_count": 1}

    class Pipeline:
        @staticmethod
        def _grounded_domain_evidence(agentic, domain_id, bundle):
            return {
                "queries": [{
                    "query": "minecraft colony persistent state source",
                    "evidence_records": [{
                        "source_id": "github:colony",
                        "source_type": "github_source",
                        "content": "persistent colony saved state implementation",
                        "content_sha256": "sha256:new",
                    }],
                    "github_provider_status": "available",
                }]
            }

    class Agentic:
        class SpecValidationError(ValueError):
            pass

        @staticmethod
        def _validate_sufficient_research(note, *, allowed_refs):
            assert note["claims"]
            assert all(ref in allowed_refs for claim in note["claims"] for ref in claim["evidence_refs"])

    note = quality._quality_research_document_domain(
        Pipeline,
        Agentic,
        Rag,
        router=object(),
        prompt="식민지를 만들어줘",
        domain={"domain_id": "req_colony", "queries": ["minecraft colony progression"]},
        document={"document_sha256": "base-doc", "page_count": 1},
        trace_metadata=None,
    )
    assert calls["forced"] == 1
    assert note["sufficient"] is True
    assert note["quality_contract"]["corrective_rounds_executed"] == 1
    assert note["quality_contract"]["donor_selection_performed"] is False
