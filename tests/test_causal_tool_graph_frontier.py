from __future__ import annotations

from minecraft_mod_ai import causal_tool_graph as graph


def _transition(
    name: str,
    *,
    requires: set[str],
    effects: set[str],
    cost: int = 1,
) -> graph.ToolTransition:
    return graph.ToolTransition(
        name=name,
        preconditions=frozenset(requires),
        effects=frozenset(effects),
        cost=cost,
    )


def test_frontier_collects_equal_first_steps_without_per_candidate_search(monkeypatch) -> None:
    transitions = {
        "alpha": _transition(
            "alpha",
            requires={"start"},
            effects={"mid"},
        ),
        "beta": _transition(
            "beta",
            requires={"start"},
            effects={"mid"},
        ),
        "finish": _transition(
            "finish",
            requires={"mid"},
            effects={"project_observed"},
        ),
        "expensive_direct": _transition(
            "expensive_direct",
            requires={"start"},
            effects={"project_observed"},
            cost=3,
        ),
    }
    monkeypatch.setattr(graph, "_transitions", lambda _schemas: transitions)

    def fail_if_tail_search_restarts(*_args, **_kwargs):
        raise AssertionError("frontier restarted a per-candidate shortest-path search")

    monkeypatch.setattr(
        graph,
        "_shortest_causal_path_from_transitions",
        fail_if_tail_search_restarts,
    )

    assert graph.executable_frontier(
        (),
        state=frozenset({"start"}),
        goals=("observe",),
        limit=3,
        max_depth=2,
        preference={"beta": 0, "alpha": 1},
    ) == ("beta", "alpha")


def test_external_goal_without_executable_path_does_not_recurse(monkeypatch) -> None:
    transitions = {
        "external_mcp_call": _transition(
            "external_mcp_call",
            requires={"external_schema"},
            effects={"external_observation"},
        )
    }
    monkeypatch.setattr(graph, "_transitions", lambda _schemas: transitions)

    assert graph.executable_frontier(
        (),
        state=frozenset({"workspace_bound"}),
        goals=("external",),
        max_depth=3,
    ) == ()
