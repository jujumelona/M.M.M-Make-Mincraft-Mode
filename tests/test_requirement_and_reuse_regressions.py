from __future__ import annotations

from typing import Any

from minecraft_mod_ai.catalog_first_grounded_rag import _query_bundle
from minecraft_mod_ai.planning_state_resolution import _normalize_requirement_rows


def test_requirement_normalization_preserves_distinct_acceptance_contracts() -> None:
    raw = {
        "requirements": [
            {
                "statement": "Open a glider from the inventory.",
                "semantic_capability": "custom",
                "acceptance": ["Opening the item shows the glider UI."],
            },
            {
                "statement": "Open a glider from the inventory.",
                "semantic_capability": "custom",
                "acceptance": ["Closing the UI returns control to the player."],
            },
        ]
    }

    rows = _normalize_requirement_rows(raw, {}, "")

    assert len(rows) == 2
    assert rows[0]["acceptance"] != rows[1]["acceptance"]


def test_requirement_normalization_collapses_only_exact_semantic_duplicates() -> None:
    row = {
        "statement": "Open a glider from the inventory.",
        "semantic_capability": "custom",
        "acceptance": ["Opening the item shows the glider UI."],
    }

    rows = _normalize_requirement_rows({"requirements": [row, dict(row)]}, {}, "")

    assert len(rows) == 1


class _CatalogBackend:
    def __init__(self) -> None:
        self.github_queries: list[str] = []

    def _search_modrinth(self, query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return (
            [
                {
                    "source_id": "modrinth:glider",
                    "url": "https://modrinth.com/mod/glider",
                    "title": "Glider",
                    "content": "A glider mod.",
                    "metadata": {"source_url": ""},
                }
            ],
            {"provider": "modrinth", "status": "available", "result_count": 1},
        )

    def _linked_github_sources(
        self,
        records: list[dict[str, Any]],
        *,
        disabled: Any,
        disable: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return [], {"provider": "github", "status": "available", "result_count": 0}

    def _search_github(
        self,
        query: str,
        *,
        disabled: Any,
        disable: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.github_queries.append(query)
        return (
            [
                {
                    "source_id": "github:example/glider",
                    "url": "https://github.com/example/glider",
                    "title": "Glider",
                    "content": "Source for a Minecraft glider mod.",
                }
            ],
            {
                "provider": "github",
                "status": "available",
                "result_count": 1,
                "search_requests": 1,
                "source_requests": 1,
            },
        )

    def _versions(self, router: Any) -> tuple[str, ...]:
        return ()

    def _sha256_text(self, query: str) -> str:
        return f"sha256:{query}"

    def _error(self, provider: str, exc: BaseException) -> dict[str, Any]:
        return {"provider": provider, "status": "error", "error": str(exc)}


def test_catalog_hit_without_source_link_falls_back_to_github_donor_discovery() -> None:
    backend = _CatalogBackend()

    bundle = _query_bundle(
        backend,
        object(),
        "minecraft glider implementation",
        ("modrinth",),
        github_disabled=lambda: False,
        disable_github=lambda: None,
    )

    sources = bundle["external_rag"]["sources"]
    provider = bundle["external_rag"]["providers"]["github"]
    assert backend.github_queries == ["minecraft glider implementation"]
    assert any(source["source_id"] == "github:example/glider" for source in sources)
    assert provider["policy"] == "catalog_candidate_source_discovery_fallback"
    assert (
        bundle["external_rag"]["provider_policy"]["github_broad_search"]
        == "fallback_after_missing_linked_source_or_empty_catalog"
    )
