from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

from minecraft_mod_ai import pre_design_domain_research as owner
from minecraft_mod_ai import pre_design_rag_quality_contract as quality


def test_direct_owner_fuses_initial_evidence_and_projects_all_quality_pages(monkeypatch, tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "grounded_rag": {
                    "domain_id": "req_colony",
                    "queries": [
                        {
                            "query": "minecraft colony progression",
                            "evidence_records": [
                                {
                                    "source_id": "github:a",
                                    "source_type": "github_source",
                                    "content": "colony progression persistent settlement",
                                    "content_sha256": "sha256:shared",
                                }
                            ],
                            "github_provider_status": "available",
                        },
                        {
                            "query": "colony persistent settlement source",
                            "evidence_records": [
                                {
                                    "source_id": "github:b",
                                    "source_type": "github_source",
                                    "content": "colony progression persistent settlement",
                                    "content_sha256": "sha256:shared",
                                }
                            ],
                            "github_provider_status": "available",
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "evidence-ledger.jsonl"
    ledger_rows = [
        {
            "page_ref": "sha256:initial#page=1/1",
            "unit_id": "initial",
            "part_index": 0,
            "part_count": 1,
            "content": "initial evidence",
        },
        {
            "page_ref": "sha256:corrective#page=1/1",
            "unit_id": "corrective",
            "part_index": 0,
            "part_count": 1,
            "content": "corrective evidence",
        },
    ]
    ledger_path.write_text(
        "".join(json.dumps(row) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )
    seen = {}

    class Rag:
        _EVIDENCE_PAGE_CHARS = 1800

        @staticmethod
        def _materialize_domain_evidence_document(domain_id, evidence):
            seen["materialized"] = evidence
            pages = tmp_path / "fused-pages.jsonl"
            pages.write_text(
                json.dumps(
                    {
                        "page_ref": "sha256:initial#page=1/1",
                        "content": "initial evidence",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return {
                "domain_id": domain_id,
                "document_sha256": "sha256:fused",
                "raw_path": str(raw_path),
                "pages_path": str(pages),
                "page_count": 1,
                "page_chars": 1800,
                "page_bytes": 1800,
            }

        @staticmethod
        def _atomic_write_text(path, content):
            Path(path).write_text(content, encoding="utf-8")

    def fake_quality(pipeline, agentic, project_rag, router, **kwargs):
        del pipeline, agentic, project_rag, router
        seen["quality_document"] = dict(kwargs["document"])
        return {
            "domain_id": "req_colony",
            "claims": [
                {
                    "claim": "Colony state persists.",
                    "evidence_refs": ["sha256:corrective#page=1/1"],
                }
            ],
            "gaps": [],
            "next_queries": [],
            "procedures": [],
            "sufficient": True,
            "evidence_ledger": {"path": str(ledger_path)},
            "checkpoint": {"checkpoint_dir": str(tmp_path)},
            "quality_contract": {"schema_version": "mmm/pre-design-rag-quality-v1"},
        }

    monkeypatch.setattr(quality, "_quality_research_document_domain", fake_quality)
    pipeline = ModuleType("minecraft_mod_ai.pre_design_research_pipeline")
    pipeline._grounded_domain_evidence = lambda *args, **kwargs: {}
    monkeypatch.setitem(__import__("sys").modules, pipeline.__name__, pipeline)

    document = {
        "domain_id": "req_colony",
        "document_sha256": "sha256:raw",
        "raw_path": str(raw_path),
        "pages_path": str(tmp_path / "raw-pages.jsonl"),
        "page_count": 1,
        "page_chars": 1800,
        "page_bytes": 1800,
    }
    note = owner.research_document_domain(
        object(),
        Rag,
        object(),
        prompt="식민지",
        domain={"domain_id": "req_colony"},
        document=document,
        trace_metadata=None,
    )

    fused = seen["materialized"]["grounded_rag"]
    assert fused["fusion"]["duplicate_record_count"] == 1
    assert len(fused["queries"][0]["evidence_records"]) == 1
    assert seen["quality_document"]["document_sha256"] == "sha256:fused"
    assert note["claims"][0]["evidence_refs"] == ["sha256:corrective#page=1/1"]
    assert document["schema_version"] == "mmm/research-evidence-validation-view-v1"
    assert document["page_count"] == 2
    refs = [
        json.loads(line)["page_ref"]
        for line in Path(document["pages_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert refs == ["sha256:initial#page=1/1", "sha256:corrective#page=1/1"]
