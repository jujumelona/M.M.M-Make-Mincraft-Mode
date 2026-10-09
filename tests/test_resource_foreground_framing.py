"""Regression coverage for foreground masks disappearing at native Minecraft size."""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from minecraft_mod_ai.resource_catalog import texture_contract
from minecraft_mod_ai.resource_image_pipeline import (
    finalize_candidate_sources,
    frame_segmented_foreground,
    validate_texture,
)


def _item_contract():
    target_path = "src/main/resources/assets/example/textures/item/crystal.png"
    resource = texture_contract(
        binding={},
        render_kind="item.generated",
        namespace="example",
        minecraft_version="1.21.1",
        target_path=target_path,
        role="layer0",
        width=16,
        height=16,
        topology="isolated_sprite",
        alpha="transparent",
    )
    return {
        "width": 16, "height": 16,
        "role": "layer0",
        "target_path": target_path,
        "alpha_policy": "transparent",
        "resource_contract": resource,
    }


def test_tiny_genuine_alpha_subject_survives_native_pixel_grid(tmp_path):
    """Before framing, nearest-neighbor could sample a 6px subject as empty."""
    texture = _item_contract()
    source = tmp_path / "source.png"
    output = tmp_path / "native.png"
    Image.new("RGB", (256, 256), "#9f9f9f").save(source, "PNG")

    def segmenter(rgb):
        rgba = rgb.convert("RGBA")
        with Image.new("L", rgb.size, 0) as mask:
            ImageDraw.Draw(mask).ellipse((118, 118, 124, 124), fill=255)
            rgba.putalpha(mask)
        return rgba

    pending = [({"name": "frame_0", "box": [0, 0, 16, 16]}, source, True)]
    evidence = finalize_candidate_sources(
        texture, pending, output=output, resolution=(256, 256),
        segment_foreground_callback=segmenter,
    )
    assert evidence["sources"][0]["region"] == "frame_0"
    assert validate_texture(output, texture)["status"] == "PASS"
    with Image.open(output) as generated:
        alpha = generated.getchannel("A")
        try:
            assert alpha.getbbox()
            assert alpha.getpixel((0, 0)) == 0
            assert alpha.getpixel((8, 8)) == 255
            assert set(alpha.getdata()) == {0, 255}
        finally:
            alpha.close()


def test_zero_confidence_matting_fails_instead_of_inventing_object():
    with Image.new("RGBA", (256, 256), (80, 100, 120, 20)) as matte:
        with pytest.raises(ValueError, match="ALPHA_MASK_BELOW_NATIVE_THRESHOLD"):
            frame_segmented_foreground(matte)


def test_non_isolated_alpha_mask_is_rejected():
    with Image.new("RGBA", (256, 256), (80, 100, 120, 255)) as matte:
        with pytest.raises(ValueError, match="ALPHA_MASK_NOT_ISOLATED"):
            frame_segmented_foreground(matte)
