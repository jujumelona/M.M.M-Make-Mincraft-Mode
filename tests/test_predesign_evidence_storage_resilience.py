from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai import pre_design_grounded_rag as rag
from minecraft_mod_ai.pre_design_research_pipeline import _validate_document_grounding


def test_unicode_line_separators_preserve_single_line_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    # External text with tricky Unicode line separator, paragraph separator, and next line
    body = "prefix \u2028 middle \u2029 suffix \x85 end"
    evidence = {
        "grounded_rag": {
            "queries": [
                {
                    "query": "unicode test",
                    "evidence_records": [
                        {
                            "source_id": "source:unicode",
                            "source_type": "test",
                            "url": "https://example.invalid/unicode",
                            "title": "unicode",
                            "content": body,
                        }
                    ],
                }
            ]
        }
    }
    document = rag._materialize_domain_evidence_document("unicode_domain", evidence)
    assert document["page_count"] == 1

    # Read from in-memory cache
    pages_mem = rag._read_evidence_pages(document)
    assert len(pages_mem) == 1
    assert body in pages_mem[0]["content"]

    # Read from disk directly (strip _pages)
    disk_doc = dict(document)
    disk_doc.pop("_pages", None)
    pages_disk = rag._read_evidence_pages(disk_doc)
    assert len(pages_disk) == 1
    assert body in pages_disk[0]["content"]


def test_corrupted_pages_jsonl_self_heals_from_raw(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    body = "important spaceship modular design facts"
    evidence = {
        "grounded_rag": {
            "queries": [
                {
                    "query": "spaceship",
                    "evidence_records": [
                        {
                            "source_id": "source:ship",
                            "source_type": "test",
                            "url": "https://example.invalid/ship",
                            "title": "ship",
                            "content": body,
                        }
                    ],
                }
            ]
        }
    }
    document = rag._materialize_domain_evidence_document("ship_domain", evidence)
    pages_path = Path(document["pages_path"])
    raw_path = Path(document["raw_path"])

    assert pages_path.is_file()
    assert raw_path.is_file()

    # Simulate truncated/corrupted pages_path (e.g. char 11 JSONDecodeError)
    pages_path.write_text('{"content": "', encoding="utf-8")

    # Read with disk document (no in-memory cache)
    disk_doc = dict(document)
    disk_doc.pop("_pages", None)

    # Must self-heal from raw_path without raising JSONDecodeError
    healed_pages = rag._read_evidence_pages(disk_doc)
    assert len(healed_pages) == 1
    assert body in healed_pages[0]["content"]

    # Disk file must now be valid JSONL
    disk_lines = [l for l in pages_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    assert len(disk_lines) == 1
    parsed = json.loads(disk_lines[0])
    assert parsed["domain_id"] == "ship_domain"


def test_unrecoverable_pages_quarantines_and_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    bad_pages = tmp_path / "bad.pages.jsonl"
    bad_raw = tmp_path / "bad.json"

    bad_pages.write_text('{"corrupted": ', encoding="utf-8")
    # No raw file exists

    doc = {
        "domain_id": "bad_domain",
        "page_count": 1,
        "pages_path": str(bad_pages),
        "raw_path": str(bad_raw),
        "document_sha256": "sha256:dummy",
    }

    # Should not crash with JSONDecodeError; must quarantine corrupt file and return []
    result = rag._read_evidence_pages(doc)
    assert result == []
    assert not bad_pages.exists()  # quarantined/unlinked


def test_validate_document_grounding_resilience(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    body = "grounded claim text"
    evidence = {
        "grounded_rag": {
            "queries": [
                {
                    "query": "q",
                    "evidence_records": [
                        {
                            "source_id": "s:1",
                            "source_type": "test",
                            "url": "https://example.invalid",
                            "title": "t",
                            "content": body,
                        }
                    ],
                }
            ]
        }
    }
    document = rag._materialize_domain_evidence_document("grounded_domain", evidence)
    page_ref = document["_pages"][0]["page_ref"]

    note_with_card = {
        "grounded_evidence_cards": [{"page_ref": page_ref}],
        "claims": [{"claim": body, "citations": [page_ref]}],
    }

    # Normal validation passes
    _validate_document_grounding(None, rag, note_with_card, document, domain_id="grounded_domain")

    # Now corrupt the disk file and remove _pages; self-healing should allow validation to still pass
    Path(document["pages_path"]).write_text('{"truncated', encoding="utf-8")
    disk_doc = dict(document)
    disk_doc.pop("_pages", None)

    _validate_document_grounding(None, rag, note_with_card, disk_doc, domain_id="grounded_domain")
