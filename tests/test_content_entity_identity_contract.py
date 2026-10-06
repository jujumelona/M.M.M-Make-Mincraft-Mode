from __future__ import annotations

import re

from minecraft_mod_ai import content_design_graph as content_graph
from minecraft_mod_ai.task_template_catalog import load_record_template


_COLLIDING_ID = "space_mode_market_trade_gui_rec_"


def test_entity_id_schema_does_not_force_32_character_prefix_collisions() -> None:
    template = load_record_template("design/content_entity")
    entity_id = template["record_schema"]["properties"]["entity_id"]

    assert entity_id["maxLength"] == 64


def test_host_deterministically_splits_cross_kind_entity_id_collisions() -> None:
    entities = {
        _COLLIDING_ID: {
            "entity_id": _COLLIDING_ID,
            "kind": "gui",
            "role": "market trading screen",
        }
    }
    node = {
        "entity_id": _COLLIDING_ID,
        "kind": "recipe",
        "role": "market trade recipe",
    }

    first = content_graph._collision_safe_entity_id(node, "req_trade_recipe", entities)
    second = content_graph._collision_safe_entity_id(node, "req_trade_recipe", entities)

    assert first == second
    assert first != _COLLIDING_ID
    assert len(first) <= 64
    assert re.fullmatch(r"[a-z][a-z0-9_]+", first)
    assert first.startswith("space_mode_market_trade_gui_rec")


def test_same_kind_reuse_keeps_the_authored_entity_id() -> None:
    entities = {
        _COLLIDING_ID: {
            "entity_id": _COLLIDING_ID,
            "kind": "gui",
            "role": "market trading screen",
        }
    }
    node = {
        "entity_id": _COLLIDING_ID,
        "kind": "gui",
        "role": "market trading screen",
    }

    assert (
        content_graph._collision_safe_entity_id(node, "req_trade_gui", entities)
        == _COLLIDING_ID
    )
