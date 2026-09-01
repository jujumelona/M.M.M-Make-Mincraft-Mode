from __future__ import annotations

from minecraft_mod_ai import pre_design_domain_research as owner


def test_direct_owner_is_small_model_host_pipeline_and_missing_receipt_helper_is_safe():
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
            return "NONE"

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

    assert owner.research_document_domain.__module__.endswith("small_model_predesign_research")
    assert len(calls) == 1
    assert calls[0]["response_format"] == "text"
    assert calls[0]["response_schema"] is None
    assert note["research_mode"] == "advisory_predesign"
    assert note["research_evidence_status"] == "no_relevant_external_evidence"
    assert note["sufficient"] is True
    assert note["gaps"] == []
    assert note["quality_contract"]["model_json"] is False
    assert note["quality_contract"]["model_corrective_queries"] is False
    assert note["evidence_document"]["document_sha256"] == "sha256:doc"
