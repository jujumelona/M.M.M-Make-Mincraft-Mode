from __future__ import annotations

from minecraft_mod_ai import authored_scope_research_contract as authored
from minecraft_mod_ai import pre_design_rag_corrective as corrective


def test_pre_design_preserves_every_approved_query_without_cap(monkeypatch) -> None:
    prompt = "우주선을 부위마다 만들고 다른 행성을 탐사하는 모드"
    planned_queries = [f"minecraft mod mechanic {index} discovery" for index in range(30)]
    monkeypatch.setattr(
        authored,
        "_active_catalog",
        lambda value: {
            "requirements": [
                {
                    "requirement_id": "req_spacecraft",
                    "search_queries": planned_queries,
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
                "providers": ["official_docs", "project_rag", "runtime"],
                "queries": [prompt],
            }
        ]
    }

    rewritten = authored._rewrite_pre_design_candidate(prompt, candidate)
    domain = rewritten["domains"][0]

    assert domain["queries"] == planned_queries
    assert len(domain["queries"]) == 30
    assert prompt not in domain["queries"]
    assert "github" in domain["providers"]
    assert "modrinth" in domain["providers"]


def test_corrective_query_uses_unsearched_host_queries_without_model_call() -> None:
    initial = "minecraft mod initial query"
    approved = [
        initial,
        "minecraft mod fabric modular spacecraft source",
        "minecraft mod fabric planet colonization implementation",
    ]

    class ExplodingRouter:
        def __getattr__(self, name):
            raise AssertionError(f"corrective query planning must not call model: {name}")

    seen = {initial}
    result = corrective._generate_gap_queries(
        object(),
        object(),
        ExplodingRouter(),
        domain={
            "domain_id": "request",
            "requirements": ["spacecraft"],
            "queries": approved,
        },
        gaps=["Initial evidence contained metadata but no substantive source."],
        prior_queries=[initial],
        seen=seen,
        raw_prompt="우주모드",
        progress_label="test",
    )

    assert result == approved[1:]
    assert seen == set(approved)
