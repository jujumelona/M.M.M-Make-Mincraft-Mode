from __future__ import annotations

from minecraft_mod_ai.external_agent_bridge import TOOL_NAMES as EXTERNAL_TOOL_NAMES
from minecraft_mod_ai.mcp_server import _TOOL_STAGES
from minecraft_mod_ai.tool_transition_registry import TRANSITIONS


def test_every_builtin_model_surface_has_reviewed_causal_transition() -> None:
    expected = set(_TOOL_STAGES) | set(EXTERNAL_TOOL_NAMES)
    missing = sorted(expected - set(TRANSITIONS))
    assert missing == [], f"Built-in MMM tools missing causal transitions: {missing}"


def test_registry_has_no_description_derived_wildcard_transition() -> None:
    assert "*" not in TRANSITIONS
    assert "default" not in TRANSITIONS
    assert all(spec.preconditions for spec in TRANSITIONS.values())
    assert all(spec.effects for spec in TRANSITIONS.values())
