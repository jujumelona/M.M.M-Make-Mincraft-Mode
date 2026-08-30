from __future__ import annotations

from minecraft_mod_ai import authored_scope_research_contract as scope
from minecraft_mod_ai import grounded_rag_runtime_contract as runtime


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


def test_batch_retrieval_preserves_caller_key_after_whitespace_canonicalization(monkeypatch):
    coordinator = runtime.GroundedRAGCoordinator()

    def fake(query, versions):
        del versions
        assert query == "spaceship component construction"
        return {
            "schema_version": "mmm/external-grounded-rag",
            "status": "available",
            "query": query,
            "providers": ["github_public_source"],
            "documents": [],
            "errors": [],
            "github_retrieval": {},
        }

    monkeypatch.setattr(runtime, "_BASE_EXTERNAL_RETRIEVAL", fake)
    original = "spaceship  component   construction"
    result = coordinator.retrieve_many((original,), ())

    assert original in result
    assert result[original]["query"] == "spaceship component construction"
    coordinator.executor.shutdown(wait=True, cancel_futures=True)
