from __future__ import annotations

from minecraft_mod_ai import content_design_graph as content_graph
from minecraft_mod_ai.content_design_contract import (
    CONTENT_KIND_TO_FACT_TYPE,
    fact_type_for_content_kind,
)


def test_content_capability_mapping_is_host_owned() -> None:
    assert not hasattr(content_graph, "_resolve_content_capabilities")
    for kind, fact_type in CONTENT_KIND_TO_FACT_TYPE.items():
        assert fact_type_for_content_kind(kind) is fact_type
