from __future__ import annotations

from minecraft_mod_ai import authored_scope_research_contract as scope


def _requirements():
    return [
        {
            "requirement_id": "req_build",
            "source_span": {"text": "우주선을 부위마다 만들어서"},
        },
        {
            "requirement_id": "req_launch",
            "source_span": {"text": "그렇게해서 우주로 나갈수있고"},
        },
    ]


def test_retrieval_plan_uses_english_queries_and_authored_dependencies():
    payload = {
        "requirements": [
            {
                "requirement_id": "req_build",
                "depends_on": [],
                "search_queries": [
                    "minecraft modular spaceship construction mod",
                    "spaceship component assembly source implementation",
                ],
            },
            {
                "requirement_id": "req_launch",
                "depends_on": ["req_build"],
                "search_queries": [
                    "minecraft spaceship launch travel mod",
                    "space travel vehicle source implementation",
                ],
            },
        ]
    }

    result = scope._normalize_retrieval_plan(
        "우주모드 만들어줘",
        _requirements(),
        payload,
    )

    assert result["req_launch"]["depends_on"] == ["req_build"]
    assert result["req_build"]["search_queries"][0].startswith("minecraft modular")
    assert all(
        any("a" <= char.casefold() <= "z" for char in query)
        for item in result.values()
        for query in item["search_queries"]
    )


def test_raw_non_english_prompt_cannot_be_the_retrieval_plan():
    payload = {
        "requirements": [
            {
                "requirement_id": "req_build",
                "depends_on": [],
                "search_queries": ["우주선을 부위마다 만들어서", "우주선 만들기"],
            },
            {
                "requirement_id": "req_launch",
                "depends_on": ["req_build"],
                "search_queries": ["그렇게해서 우주로 나갈수있고", "우주 이동"],
            },
        ]
    }

    try:
        scope._normalize_retrieval_plan(
            "우주선을 부위마다 만들어서 그렇게해서 우주로 나갈수있고",
            _requirements(),
            payload,
        )
    except ValueError as exc:
        assert "English queries" in str(exc)
    else:
        raise AssertionError("raw authored text leaked into external retrieval queries")


def test_approved_queries_replace_raw_request_and_enable_public_research_routes(monkeypatch):
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
                "providers": ["official_docs", "project_rag", "external_mcp"],
                "queries": [prompt],
            }
        ]
    }

    rewritten = scope._rewrite_pre_design_candidate(prompt, candidate)
    domain = rewritten["domains"][0]

    assert prompt not in domain["queries"]
    assert domain["queries"][0] == "minecraft modular spaceship construction mod"
    assert {"github", "modrinth"} <= set(domain["providers"])
