from __future__ import annotations

from minecraft_mod_ai import pre_design_grounded_rag as rag


def test_materialization_keeps_large_source_body_as_one_lossless_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    body = "source-body-" + ("z" * 5000)
    evidence = {"grounded_rag": {"queries": [{"query": "large source", "evidence_records": [{"source_id": "source:large", "source_type": "test", "url": "https://example.invalid/source", "title": "large", "content": body}]}]}}
    document = rag._materialize_domain_evidence_document("domain", evidence)
    pages = rag._read_evidence_pages(document)
    assert document["page_count"] == document["model_unit_count"] == 1
    assert document["page_partition"] == "claim_bearing_source_unit"
    assert len(pages) == 1
    assert body in pages[0]["content"]
    assert pages[0]["part_count"] == 1
