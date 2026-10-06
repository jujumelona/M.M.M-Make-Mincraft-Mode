from __future__ import annotations

from minecraft_mod_ai.complete_spec import AssetRequest
from minecraft_mod_ai.module_identity import logical_module_id
from minecraft_mod_ai.spec_identity import (
    SPEC_ID_MAX_LENGTH,
    SPEC_ID_RE,
    canonical_spec_id,
)


def test_failing_long_content_asset_id_is_bounded_deterministically() -> None:
    raw = (
        "texture_item_"
        "alien_faction_mineral_node_loot_table_space_mode_engineering_b1a"
    )

    first = canonical_spec_id(raw)
    second = canonical_spec_id(raw)

    assert first == second
    assert len(first) <= SPEC_ID_MAX_LENGTH
    assert SPEC_ID_RE.fullmatch(first)
    assert first != raw
    assert first.startswith("texture_item_alien_faction_mineral_node")


def test_long_ids_with_same_prefix_do_not_collapse_to_same_identity() -> None:
    shared = "texture_item_" + ("a" * 90)

    left = canonical_spec_id(shared + "_left")
    right = canonical_spec_id(shared + "_right")

    assert left != right
    assert len(left) <= SPEC_ID_MAX_LENGTH
    assert len(right) <= SPEC_ID_MAX_LENGTH
    assert SPEC_ID_RE.fullmatch(left)
    assert SPEC_ID_RE.fullmatch(right)


def test_asset_validator_accepts_canonicalized_derived_id() -> None:
    asset_id = canonical_spec_id(
        "texture_item_"
        "alien_faction_mineral_node_loot_table_space_mode_engineering_b1a"
    )
    request = AssetRequest(
        asset_id=asset_id,
        kind="item",
        visual_description="angular alien mineral node, metallic crystalline surface",
        render_kind="item.generated",
        subject_id="alien_faction_mineral_node",
        owner_module_id="alien_faction_mineral_node",
    )

    request.validate()


def test_long_logical_module_ids_use_same_collision_safe_contract() -> None:
    left = logical_module_id(
        ":" + ("very-long-module-segment-" * 5) + "left"
    )
    right = logical_module_id(
        ":" + ("very-long-module-segment-" * 5) + "right"
    )

    assert left != right
    assert len(left) <= SPEC_ID_MAX_LENGTH
    assert len(right) <= SPEC_ID_MAX_LENGTH
    assert SPEC_ID_RE.fullmatch(left)
    assert SPEC_ID_RE.fullmatch(right)
