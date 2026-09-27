from __future__ import annotations

from minecraft_mod_ai import reuse_planner


def test_static_capability_graph_has_no_fixed_first_six_cap() -> None:
    capabilities = tuple(f"capability_{index}" for index in range(12))

    graph = reuse_planner.decompose_capability_graph(
        "approved static capabilities",
        module_kinds=capabilities,
    )

    assert graph.nodes == capabilities
    assert len(graph.nodes) == 12


def test_legacy_runtime_donor_inspection_surface_is_retired() -> None:
    assert not hasattr(reuse_planner, "inspect_repository_slice")
    assert not hasattr(reuse_planner, "_discover_best_donor")
