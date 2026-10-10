"""Reproduce the explicit hardness/resistance silently lost by the 26.2 debug mod."""
import pytest

from minecraft_mod_ai.block_strength_lowering import (
    authored_block_strength,
    lower_block_strength,
)


def test_authored_block_settings_are_lowered_into_compilable_java():
    strength = authored_block_strength({"display_name": "Crystal Block", "hardness": 2.0, "resistance": 3.0})
    assert strength == (2.0, 3.0)
    raw = (
        "new Block(BlockBehaviour.Properties.of()"
        ".setId(ModBlockIds.CRYSTAL_BLOCK_KEY)"
        "/* MMM:block_properties:crystal_block */)"
    )
    result = lower_block_strength(raw, "crystal_block", strength)
    assert ".strength(2F, 3F)" in result
    assert "/* MMM:block_properties:" not in result


@pytest.mark.parametrize("config", [
    {"hardness": 2.0},
    {"resistance": 3.0},
    {"hardness": float("nan"), "resistance": 3.0},
    {"hardness": True, "resistance": 3.0},
    {"hardness": 2.0, "resistance": -1},
    {"hardness": "2.0", "resistance": 3},
])
def test_invalid_authored_block_properties_fail_closed(config):
    with pytest.raises(ValueError, match="BLOCK_PROPERT"):
        authored_block_strength(config)


def test_unbound_marker_cannot_write_to_different_block():
    with pytest.raises(ValueError, match="BLOCK_PROPERTIES_ANCHOR_MISMATCH"):
        lower_block_strength(
            "/* MMM:block_properties:other_block */",
            "crystal_block",
            (2.0, 3.0),
        )


def test_blocks_without_strength_settings_keep_default_minecraft_behavior():
    assert authored_block_strength({"display_name": "Example"}) is None
