from __future__ import annotations

import pytest

from minecraft_mod_ai.authored_content_contract import (
    CONTENT_CONCERN_MINIMUM_ENTITY_COUNT,
)
from minecraft_mod_ai.content_design_graph import (
    SlotFillError,
    _prune_optional_orphan_resource_entities,
)


def _node(kind: str, *requirement_refs: str) -> dict[str, object]:
    return {
        "kind": kind,
        "requirement_refs": list(requirement_refs),
    }


def _owned(requirement_id: str, minimum: int) -> dict[str, object]:
    return {
        "requirement_id": requirement_id,
        "minimum_entity_count": minimum,
    }


def test_engineering_resource_concerns_do_not_force_content_identity() -> None:
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["registries"] == 0
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["data_resources"] == 0
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["assets"] == 1
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["interactions"] == 1
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["displayed_state"] == 1


def test_optional_orphan_registry_tag_is_left_for_platform_coverage() -> None:
    entities = {
        "ore_tags": _node("registry_tag", "resource_data"),
    }

    removed = _prune_optional_orphan_resource_entities(
        entities,
        [_owned("resource_data", 0)],
    )

    assert removed == ("ore_tags",)
    assert entities == {}


def test_required_orphan_registry_tag_fails_closed() -> None:
    entities = {
        "required_tag": _node("registry_tag", "gameplay"),
    }

    with pytest.raises(
        SlotFillError,
        match="CONTENT_RESOURCE_TARGET_UNRESOLVED: required_tag: registry_tag",
    ):
        _prune_optional_orphan_resource_entities(
            entities,
            [_owned("gameplay", 1)],
        )


def test_registry_tag_with_concrete_member_survives() -> None:
    entities = {
        "ore_tags": _node("registry_tag", "resource_data"),
        "moon_ore": _node("item", "gameplay"),
    }

    removed = _prune_optional_orphan_resource_entities(
        entities,
        [_owned("resource_data", 0), _owned("gameplay", 1)],
    )

    assert removed == ()
    assert set(entities) == {"ore_tags", "moon_ore"}


def test_recipe_with_concrete_item_survives() -> None:
    entities = {
        "alloy_recipe": _node("crafting_recipe", "resource_data"),
        "alloy": _node("item", "gameplay"),
    }

    removed = _prune_optional_orphan_resource_entities(
        entities,
        [_owned("resource_data", 0), _owned("gameplay", 1)],
    )

    assert removed == ()
    assert set(entities) == {"alloy_recipe", "alloy"}
