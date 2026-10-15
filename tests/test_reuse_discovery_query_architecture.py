from __future__ import annotations

from minecraft_mod_ai import reuse_discovery


def test_source_discovery_does_not_filter_by_binary_release_target(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "1")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    class Client:
        def __init__(self):
            self.calls = []

        def search(self, provider, query, **kwargs):
            self.calls.append(kwargs)
            return {"candidates": []}

    client = Client()
    reuse_discovery.discover_repositories_for_graph(
        ("space.travel",), client, minecraft_version="1.21.11", loader="fabric",
    )
    assert client.calls
    assert all("minecraft_version" not in call and "loader" not in call for call in client.calls)


def test_canonical_capability_query_precedes_long_authored_clause() -> None:
    variants = reuse_discovery._query_variants(
        "space.travel",
        (
            "우주선을 완성하고 성능을 업그레이드한 뒤 다른 행성으로 이동하여 여러 상호작용을 수행하는 전체 게임 시스템 구현",
        ),
    )

    assert variants[0] == "minecraft space travel spaceship mod"
    assert len(variants) <= reuse_discovery._query_variant_limit()


def test_authored_feature_query_survives_ontology_alias_budget():
    variants = reuse_discovery._query_variants("space travel", ("minecraft space travel mod", "space travel"))
    assert variants[0] == "minecraft space travel mod"
    crew = reuse_discovery._query_variants("crew recruitment", ("minecraft crew recruitment mod",))
    assert "crew recruitment" in crew[0]


def test_compound_feature_broadens_to_actual_source_nouns(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "3")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    class Client:
        def __init__(self):
            self.calls = []

        def search(self, provider, query, **kwargs):
            self.calls.append((provider, query))
            if provider == "github" and query == "ship":
                return {"candidates": [
                    {"repository": "example/ships", "title": "Minecraft mod",
                     "summary": "Build and sail physical ships"},
                    {"repository": "example/ui", "title": "Minecraft Fabric mod UI widgets"},
                ]}
            return {"candidates": []}

    client = Client()
    result = reuse_discovery.discover_repositories_for_graph(
        ("spaceship construction",), client,
        capability_graph={"search_terms": [{"capability": "spaceship construction",
                                           "terms": ["spaceship construction"]}]},
    )
    assert result["spaceship construction"] == ("example/ships",)
    assert ("github", "ship") in client.calls
    assert len(client.calls) == len(set(client.calls))


def test_catalogue_plural_and_compound_words_rank_as_related_source():
    source = {"title": "Minecraft mod", "summary": "Build controllable airships and ships"}
    assert reuse_discovery._candidate_relevance(source, "spaceship construction") > 0
    assert reuse_discovery._candidate_relevance(source, "alien combat") == 0
    unrelated = {"title": "Minecraft Fabric mod", "summary": "UI account membership and ownership"}
    assert reuse_discovery._candidate_relevance(unrelated, "spaceship construction") == 0


def test_queries_are_deduplicated_after_provider_normalization(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "3")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    class Client:
        def __init__(self):
            self.calls = []

        def search(self, provider, query, **kwargs):
            self.calls.append((provider, query))
            return {"candidates": []}

    client = Client()
    reuse_discovery.discover_repositories_for_graph(("planet ore generation",), client)
    assert len(client.calls) == len(set(client.calls))


def test_unrelated_mod_cannot_claim_feature_coverage(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "1")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    class Client:
        def search(self, provider, query, **kwargs):
            if provider != "github":
                return {"candidates": []}
            return {"candidates": [
                {"repository": "example/ui-lib", "title": "Minecraft Fabric mod UI widgets"},
                {"repository": "example/planet", "title": "Minecraft Fabric planet travel mod"},
            ]}

    result = reuse_discovery.discover_repositories_for_graph(("space travel",), Client())
    assert result["space travel"] == ("example/planet",)


def test_catalogue_membership_does_not_replace_feature_relevance(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "1")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    class Client:
        def search(self, provider, query, **kwargs):
            if provider != "modrinth":
                return {"candidates": []}
            return {"candidates": [
                {"repository": "example/ui", "title": "Minecraft UI widgets"},
                {"repository": "example/space", "title": "Travel to planets in space"},
            ]}

    result = reuse_discovery.discover_repositories_for_graph(("space travel",), Client())
    assert result["space travel"] == ("example/space",)


def test_source_project_body_recovers_relevance_missing_from_short_description(monkeypatch):
    import httpx

    def respond(request):
        return httpx.Response(200, json={
            "source_url": "https://github.com/example/eureka",
            "description": "Assemble and sail your creations",
            "body": "Build physical ships from blocks. " + "Supporting documentation. " * 400,
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(reuse_discovery, "_pooled_http_client", lambda: http)
        values = reuse_discovery._resolve_modrinth_candidates([
            {"title": "Eureka", "api_url": "https://api.modrinth.com/v2/project/eureka"},
        ], with_metadata=True)
    assert values[0][0] == "example/eureka"
    assert reuse_discovery._candidate_relevance(values[0][2], "spaceship construction") > 0
    assert len(values[0][2]["body"]) <= 6000


def test_shared_query_is_ranked_separately_for_each_feature(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "1")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    monkeypatch.setattr(reuse_discovery, "_query_variants", lambda *args: ("shared query",))

    class Client:
        def __init__(self):
            self.calls = []

        def search(self, provider, query, **kwargs):
            self.calls.append((provider, query))
            return {"candidates": [
                {"repository": "example/ships", "title": "Minecraft mod spaceship construction"},
                {"repository": "example/planets", "title": "Minecraft mod planet exploration"},
            ]} if provider == "github" else {"candidates": []}

    client = Client()
    result = reuse_discovery.discover_repositories_for_graph(("spaceship construction", "planet exploration"), client)
    assert result["spaceship construction"] == ("example/ships",)
    assert result["planet exploration"] == ("example/planets",)
    assert len(client.calls) == len(set(client.calls))


def test_next_page_keeps_a_relevant_source_after_unrelated_first_hits(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "1")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    class Client:
        def search(self, provider, query, **kwargs):
            if provider != "github":
                return {"candidates": []}
            if kwargs.get("cursor") == "next":
                return {"candidates": [{"repository": "example/travel", "title": "Minecraft planet travel mod"}]}
            return {"candidates": [{"repository": "example/ui", "title": "Minecraft widget mod"}],
                    "next_cursor": "next"}

    result = reuse_discovery.discover_repositories_for_graph(("space travel",), Client())
    assert result["space travel"] == ("example/travel",)


def test_later_page_throttling_preserves_verified_first_page_candidates(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "1")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    class Client:
        def search(self, provider, query, **kwargs):
            if provider != "github":
                return {"candidates": []}
            if kwargs.get("cursor"):
                raise RuntimeError("provider rate limit")
            return {"candidates": [{"repository": "example/travel", "title": "Minecraft planet travel mod"}],
                    "next_cursor": "next"}

    result = reuse_discovery.discover_repositories_for_graph(("space travel",), Client())
    assert result["space travel"] == ("example/travel",)


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
                        {
                            "source_url": "https://github.com/example/space-minecraft-mod",
                            "title": "Minecraft Fabric mod for space travel",
                        }
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

    assert result["space.travel"] == ("example/space-minecraft-mod",)
    github_queries = [query for provider, query in client.calls if provider == "github"]
    assert github_queries[0] == "space travel"


def test_second_query_preserved_after_three_unverified_first_query_hits(monkeypatch):
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "2")
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    monkeypatch.setattr(reuse_discovery, "_query_variants",
                        lambda *args: ("first", "second"))

    class Client:
        def search(self, provider, query, **kwargs):
            if provider != "github":
                return {"candidates": []}
            if query == "first":
                return {"candidates": [
                    {"repository": f"example/planet{index}",
                     "title": "Minecraft planet exploration mod"}
                    for index in range(3)
                ]}
            if query == "second":
                return {"candidates": [
                    {"repository": "example/working-donor",
                     "title": "Minecraft planet exploration mod"}
                ]}
            return {"candidates": []}

    result = reuse_discovery.discover_repositories_for_graph(
        ("planet exploration",), Client(),
    )
    assert "example/working-donor" in result["planet exploration"]
    assert len(result["planet exploration"]) == 4
