from __future__ import annotations

from minecraft_mod_ai import reuse_discovery


def test_canonical_capability_query_precedes_long_authored_clause() -> None:
    variants = reuse_discovery._query_variants(
        "space.travel",
        (
            "우주선을 완성하고 성능을 업그레이드한 뒤 다른 행성으로 이동하여 여러 상호작용을 수행하는 전체 게임 시스템 구현",
        ),
    )

    assert variants[0] == "minecraft space travel spaceship mod"
    assert len(variants) <= reuse_discovery._query_variant_limit()


def test_custom_capability_gets_compact_ecosystem_queries() -> None:
    variants = reuse_discovery._query_variants(
        "interstellar_ship_construction",
        ("very long authored semantic statement that should not own provider search phrasing",),
    )

    joined = "\n".join(variants).casefold()
    assert "interstellar ship construction" in joined
    assert "mod" in joined
    assert "very long authored semantic statement" not in variants[0].casefold()


def test_github_query_removes_duplicate_provider_boilerplate() -> None:
    assert (
        reuse_discovery._provider_query(
            "github", "minecraft space travel spaceship mod"
        )
        == "space travel"
    )
    assert (
        reuse_discovery._provider_query(
            "modrinth", "minecraft space travel spaceship mod"
        )
        == "minecraft space travel spaceship mod"
    )


def test_provider_result_from_canonical_first_query_reaches_donor_frontier(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def search(self, provider, query, *, limit, target_profile):
            assert limit >= 4
            assert target_profile == "minecraft_mod"
            self.calls.append((provider, query))
            if provider == "github" and query == "space travel":
                return {
                    "candidates": [
                        {"source_url": "https://github.com/example/space-donor"}
                    ]
                }
            return {"candidates": []}

    client = Client()
    result = reuse_discovery.discover_repositories_for_graph(
        ("space.travel",),
        client,
        capability_graph={
            "search_terms": [
                {
                    "capability": "space.travel",
                    "terms": [
                        "a long source-clause rendering that is useful as fallback evidence but should not displace the canonical ecosystem query"
                    ],
                }
            ]
        },
    )

    assert result["space.travel"] == ("example/space-donor",)
    github_queries = [query for provider, query in client.calls if provider == "github"]
    assert github_queries[0] == "space travel"
