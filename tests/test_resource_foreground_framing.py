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


def _aliased_segmented_matte():
    """Genuine source-alpha columns fall strictly between native sample points."""
    source = Image.new("RGBA", (256, 256), (20, 20, 20, 0))
    for column, x in enumerate(range(16, 256, 32)):
        color = (230, 20, 40, 255) if column % 2 else (20, 150, 250, 255)
        for y in range(32, 224):
            for xx in (x, x + 1):
                source.putpixel((xx, y), color)
    return source


def test_real_thin_matte_pixels_survive_native_sampling_without_fabrication(tmp_path):
    from minecraft_mod_ai.resource_image_pipeline import postprocess_region

    texture = _item_contract()
    with _aliased_segmented_matte() as source:
        with postprocess_region(source, texture["resource_contract"], (16, 16)) as old:
            with old.getchannel("A") as alpha:
                assert alpha.getbbox() is None  # Deterministic original failure.
        with postprocess_region(
            source, texture["resource_contract"], (16, 16),
            preserve_alpha_coverage=True,
        ) as recovered:
            output = tmp_path / "recovered.png"
            recovered.save(output, "PNG")
            with recovered.getchannel("A") as alpha:
                assert alpha.getbbox() is not None
                assert alpha.getpixel((0, 0)) == 0
                assert set(alpha.getdata()) == {0, 255}
            assert validate_texture(output, texture)["status"] == "PASS"


def test_invalid_or_low_confidence_masks_are_not_repaired_into_objects(tmp_path):
    from minecraft_mod_ai.resource_image_pipeline import postprocess_region

    texture = _item_contract()
    with Image.new("RGBA", (256, 256), (80, 90, 100, 0)) as empty:
        for alpha in (0, 60):
            with Image.new("RGBA", empty.size, (80, 90, 100, alpha)) as source:
                with postprocess_region(
                    source, texture["resource_contract"], (16, 16),
                    preserve_alpha_coverage=True,
                ) as recovered:
                    target = tmp_path / f"invalid-{alpha}.png"
                    recovered.save(target)
                    with pytest.raises(ValueError, match="Invalid alpha/pixel-grid coverage"):
                        validate_texture(target, texture)


def test_finalize_recovers_real_mask_without_second_image_generation(
    monkeypatch, tmp_path,
):
    from minecraft_mod_ai import resource_image_pipeline as pipeline

    texture = _item_contract()
    source = tmp_path / "source.png"
    output = tmp_path / "native.png"
    with Image.new("RGB", (256, 256), "grey") as original:
        original.save(source, "PNG")

    # Keep the known alias pattern aligned for this deterministic integration
    # check. The framing geometry itself has separate tests above.
    monkeypatch.setattr(
        pipeline, "frame_segmented_foreground",
        lambda matte, *, alpha_threshold: matte.copy(),
    )

    def segmenter(rgb):
        with _aliased_segmented_matte() as matte:
            return matte.copy()

    receipt = pipeline.finalize_candidate_sources(
        texture,
        [({"name": "frame_0", "box": [0, 0, 16, 16]}, source, True)],
        output=output,
        resolution=(256, 256),
        segment_foreground_callback=segmenter,
    )
    assert validate_texture(output, texture)["status"] == "PASS"
    assert len(receipt["sources"]) == 1
