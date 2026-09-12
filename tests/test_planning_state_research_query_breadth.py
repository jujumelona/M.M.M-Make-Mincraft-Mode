from __future__ import annotations

from minecraft_mod_ai.planning_state_research import _compile_pending_queries


def test_compile_pending_queries_preserves_all_explicit_unique_queries() -> None:
    state = {
        "research_queue": [
            {
                "status": "pending",
                "queries": [
                    "query one",
                    "query two",
                    "query three",
                    "query four",
                    "query five",
                    "query six",
                    "query two",
                ],
            }
        ]
    }

    _compile_pending_queries(object(), state)

    assert state["research_queue"][0]["queries"] == [
        "query one",
        "query two",
        "query three",
        "query four",
        "query five",
        "query six",
    ]
