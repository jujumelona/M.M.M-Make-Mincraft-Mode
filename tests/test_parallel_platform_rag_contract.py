from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.platform_planning_contract import _target_retrieve


class _Index:
    def __init__(self, *, documents) -> None:
        self.documents = documents

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def retrieve(
        self,
        query: str,
        *,
        minecraft_version: str,
        loader: str,
        mappings: str,
        limit: int,
    ):
        return {
            "query": query,
            "minecraft_version": minecraft_version,
            "loader": loader,
            "mappings": mappings,
            "limit": limit,
        }


def test_selected_target_rag_never_falls_back_to_historical_target() -> None:
    adapter = adapter_for_target("1.21.1", "fabric")
    retrieval = SimpleNamespace(
        BUILTIN_CORPUS=(),
        OfficialCorpusIndex=_Index,
    )

    result = _target_retrieve(
        retrieval,
        "right click item interaction",
        adapter=adapter,
        limit=8,
    )

    assert result["minecraft_version"] == adapter.minecraft_version
    assert result["loader"] == adapter.loader
    assert result["mappings"] == adapter.yarn_mappings
    assert result["minecraft_version"] != "1.20.1"
