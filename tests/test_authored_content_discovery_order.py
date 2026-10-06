from __future__ import annotations

from minecraft_mod_ai.authored_content_contract import CONTENT_GRAPH_DRIVER_CONCERNS


def test_registry_binding_runs_after_concrete_content_discovery() -> None:
    assert CONTENT_GRAPH_DRIVER_CONCERNS == (
        "data_resources",
        "assets",
        "interactions",
        "displayed_state",
        "registries",
    )
    assert CONTENT_GRAPH_DRIVER_CONCERNS[-1] == "registries"
