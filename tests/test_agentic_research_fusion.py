from __future__ import annotations

import threading
from dataclasses import dataclass

from minecraft_mod_ai import central_research
from minecraft_mod_ai.agentic_research_fusion import retrieve_target_agentic_evidence


@dataclass(frozen=True)
class _Hit:
    document_id: str
    evidence_id: str
    content_sha256: str
    url: str

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "evidence_id": self.evidence_id,
            "content_sha256": self.content_sha256,
            "url": self.url,
        }


class _Receipt:
    def __init__(
        self,
        query: str,
        *,
        quality: str = "strong",
        coverage: float = 1.0,
        correction_queries: tuple[str, ...] = (),
        document_id: str = "fabric-api",
    ) -> None:
        self.query = query
        self.quality = quality
        self.coverage = coverage
        self.correction_queries = correction_queries
        self.correction_required = bool(correction_queries)
        self.hits = (
            _Hit(
                document_id=document_id,
                evidence_id=f"evidence:{query}",
                content_sha256=f"content:{document_id}",
                url=f"https://example.invalid/{document_id}",
            ),
        )

    def to_dict(self):
        return {
            "query": self.query,
            "quality": self.quality,
            "coverage": self.coverage,
            "correction_required": self.correction_required,
            "correction_queries": list(self.correction_queries),
            "hits": [item.to_dict() for item in self.hits],
        }


def _brief() -> dict:
    candidate = {
        "summary": "Parallel agentic RAG test.",
        "domains": [
            {
                "domain_id": "api",
                "objective": "Resolve API facts.",
                "requirements": ["API compatibility"],
                "evidence_kinds": ["minecraft_api", "compatibility"],
                "queries": ["primary api", "primary registry"],
                "providers": ["official_docs"],
                "depends_on": [],
            },
            {
                "domain_id": "system",
                "objective": "Resolve dependent system facts.",
                "requirements": ["System behavior"],
                "evidence_kinds": ["minecraft_api", "testing"],
                "queries": ["primary system"],
                "providers": ["official_docs"],
                "depends_on": ["api"],
            },
        ],
        "unresolved_questions": [],
    }
    return central_research.normalize_research_brief("test request", {}, candidate)


def test_agentic_target_rag_parallel_fusion_correction_and_graph_context(monkeypatch) -> None:
    monkeypatch.setenv("MMM_RESEARCH_WORKERS", "4")
    active = 0
    max_active = 0
    lock = threading.Lock()
    gate = threading.Barrier(3)

    def retrieve(
        query: str,
        *,
        minecraft_version: str,
        loader: str,
        mappings: str,
        limit: int,
    ):
        nonlocal active, max_active
        assert minecraft_version == "1.21.1"
        assert loader == "fabric"
        assert mappings
        if query.startswith("primary "):
            with lock:
                active += 1
                max_active = max(max_active, active)
            gate.wait(timeout=2)
            with lock:
                active -= 1
        if query == "primary api":
            return _Receipt(
                query,
                quality="weak",
                coverage=0.2,
                correction_queries=("api correction",),
                document_id="shared-doc",
            )
        if query == "api correction":
            return _Receipt(query, document_id="corrected-doc")
        if query == "primary registry":
            return _Receipt(query, document_id="shared-doc")
        return _Receipt(query, document_id="system-doc")

    graph = retrieve_target_agentic_evidence(
        _brief(),
        central_module=central_research,
        retrieve=retrieve,
        minecraft_version="1.21.1",
        loader="fabric",
        mappings="yarn-current",
    )

    assert max_active >= 2
    assert graph["agentic_research"]["parallel"] is True
    assert graph["agentic_research"]["primary_jobs"] == 3
    assert graph["agentic_research"]["correction_jobs"] == 1

    api = next(item for item in graph["domains"] if item["domain_id"] == "api")
    assert api["strategy"] == "agentic_adaptive_parallel"
    fused_ids = [item["document_id"] for item in api["fusion"]["documents"]]
    assert fused_ids.count("shared-doc") == 1
    assert "corrected-doc" in fused_ids
    assert api["queries"][0]["strategy"] == "corrective_multi_hop"

    system = next(item for item in graph["domains"] if item["domain_id"] == "system")
    assert system["dependency_evidence"][0]["domain_id"] == "api"
    assert "shared-doc" in system["dependency_evidence"][0]["document_ids"]


def test_agentic_target_rag_skips_non_official_domains() -> None:
    brief = central_research.normalize_research_brief(
        "source lookup",
        {},
        {
            "summary": "External-only test.",
            "domains": [
                {
                    "domain_id": "source",
                    "objective": "Find source.",
                    "requirements": ["Source evidence"],
                    "evidence_kinds": ["source_code"],
                    "queries": ["source query"],
                    "providers": ["github"],
                    "depends_on": [],
                }
            ],
            "unresolved_questions": [],
        },
    )

    def fail(*args, **kwargs):
        raise AssertionError("official retrieval must not run")

    graph = retrieve_target_agentic_evidence(
        brief,
        central_module=central_research,
        retrieve=fail,
        minecraft_version="1.21.1",
        loader="fabric",
        mappings="yarn-current",
    )

    assert graph["agentic_research"]["primary_jobs"] == 0
    assert graph["domains"][0]["retrieval_decision"] == "skip_official_lane"
