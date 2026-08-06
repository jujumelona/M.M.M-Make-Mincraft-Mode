from __future__ import annotations

from minecraft_mod_ai.mod_development_methods import (
    resolve_mod_development_methods,
)


def test_explicit_map_exclusion_is_not_treated_as_map_request() -> None:
    result = resolve_mod_development_methods(
        "월드 ZIP은 만들지 말고 음식 아이템 모드만 만들어줘"
    )

    assert result["standalone_map_requested"] is False
    assert result["standalone_map_generation"] is False
    assert "content_registry" in result["method_ids"]
    assert "fabric_worldgen" not in result["method_ids"]


def test_english_map_exclusion_is_not_treated_as_map_request() -> None:
    result = resolve_mod_development_methods(
        "Create a food mod without a standalone map or world zip."
    )

    assert result["standalone_map_requested"] is False
    assert result["standalone_map_generation"] is False
    assert "content_registry" in result["method_ids"]
