from __future__ import annotations

from minecraft_mod_ai import pre_design_domain_research as owner


def test_direct_owner_rejects_irrelevant_materialized_bodies_without_model_call():
    calls: list[dict[str, object]] = []

    class Rag:
        @staticmethod
        def _read_evidence_pages(document):
            del document
            return [
                {
                    "page_ref": "sha256:noise#page=1/1",
                    "content": "Unrelated finance dashboard material.",
                }
            ]

    class Router:
        def generate_text(self, role, messages, **kwargs):
            calls.append({"role": role, "messages": messages, **kwargs})
            raise AssertionError("host evidence projection must not call the model")

    document = {
        "domain_id": "req_colony",
        "document_sha256": "sha256:doc",
        "page_count": 1,
    }
    note = owner.research_document_domain(
        object(),
        Rag,
        Router(),
        prompt="식민지",
        domain={
            "domain_id": "req_colony",
            "objective": "persistent colony mechanics",
            "queries": ["minecraft persistent colony mechanics"],
        },
        document=document,
        trace_metadata=None,
    )

    assert owner.research_document_domain.__module__ == "minecraft_mod_ai.pre_design_domain_research"
    assert calls == []
    assert note["model_called"] is False
    assert note["source_body_count"] == 1
    assert note["host_grounded_evidence_card_count"] == 0
    assert note["research_mode"] == "advisory_predesign"
    assert note["research_evidence_status"] == "no_relevant_external_evidence"
    assert note["evidence_extraction_status"] == "no_claim_bearing_source_body"
    assert note["sufficient"] is False
    assert note["checkpoint"]["status"] == "blocked"
    assert note["research_failures"] == ["no_claim_bearing_source_body"]
    assert note["gaps"]
