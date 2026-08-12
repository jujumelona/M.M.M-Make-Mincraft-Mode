from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import central_research
from minecraft_mod_ai.parallel_platform_rag_contract import (
    _target_parallel_retrieve_factory,
)
from minecraft_mod_ai.platform_catalog import adapter_for_target


class _Receipt:
    correction_queries = ()
    hits = (object(),)
    correction_required = False

    def __init__(self, *, query: str, version: str, mappings: str) -> None:
        self.query = query
        self.version = version
        self.mappings = mappings

    def to_dict(self):
        return {
            "query": self.query,
            "minecraft_version": self.version,
            "mappings": self.mappings,
            "hits": [{"document_id": "fake"}],
        }


def test_target_parallel_rag_never_falls_back_to_1201_after_selection() -> None:
    adapter = adapter_for_target("1.21.1", "fabric")
    brief = central_research.normalize_research_brief(
        "Add one simple item with a right-click interaction.",
        {},
    )
    brief = {
        **brief,
        "_mmm_platform_target": {
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
        },
    }
    calls: list[tuple[str, str, str]] = []

    def retrieve(query: str, *, minecraft_version: str, loader: str, mappings: str, limit: int):
        calls.append((minecraft_version, loader, mappings))
        return _Receipt(query=query, version=minecraft_version, mappings=mappings)

    def legacy(*args, **kwargs):
        raise AssertionError("selected target must not use legacy RAG")

    fn = _target_parallel_retrieve_factory(
        central_module=central_research,
        retrieval_module=SimpleNamespace(retrieve_official_evidence=retrieve),
        legacy_retrieve=legacy,
    )
    result = fn(brief, retrieve=retrieve)

    assert calls
    assert all(
        value == (adapter.minecraft_version, adapter.loader, adapter.yarn_mappings)
        for value in calls
    )
    assert result["target"] == {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
    }
