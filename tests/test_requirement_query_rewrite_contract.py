from __future__ import annotations

from minecraft_mod_ai import authored_scope_research_contract as scope


def test_catalog_queries_keep_only_bounded_english_host_queries() -> None:
    catalog = {
        "requirements": [
            {
                "requirement_id": "req_build",
                "search_queries": [
                    "minecraft modular spaceship construction mod",
                    "우주선을 부위마다 만들어서",
                    "spaceship component assembly source implementation",
                ],
            },
            {
                "requirement_id": "req_launch",
                "search_queries": [
                    "minecraft spaceship launch travel mod",
                    "우주 이동",
                ],
            },
        ]
    }

    assert scope._catalog_queries(catalog) == [
        "minecraft modular spaceship construction mod",
        "spaceship component assembly source implementation",
        "minecraft spaceship launch travel mod",
    ]


def test_approved_queries_replace_raw_request_without_exposing_github_as_catalog_peer(
    monkeypatch,
) -> None:
    prompt = "우주선을 부위마다 만들어서 우주로 나가게 해줘"
    monkeypatch.setattr(
        scope,
        "_active_catalog",
        lambda value: {
            "requirements": [
                {
                    "requirement_id": "req_build",
                    "search_queries": [
                        "minecraft modular spaceship construction mod",
                        "spaceship component assembly source implementation",
                    ],
                }
            ]
        }
        if value == prompt
        else None,
    )
    candidate = {
        "domains": [
            {
                "domain_id": "request",
                "providers": ["official_docs", "project_rag", "github"],
                "queries": [prompt],
            }
        ]
    }

    rewritten = scope._rewrite_pre_design_candidate(prompt, candidate)
    domain = rewritten["domains"][0]

    assert prompt not in domain["queries"]
    assert domain["queries"][0] == "minecraft modular spaceship construction mod"
    assert {"curseforge", "modrinth"} <= set(domain["providers"])
    assert "github" not in domain["providers"]
