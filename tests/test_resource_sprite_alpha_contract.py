from __future__ import annotations

import pytest
from PIL import Image

from minecraft_mod_ai.complete_spec import AssetRequest
from minecraft_mod_ai.resource_contracts import resolve_asset
from minecraft_mod_ai.resource_image_pipeline import (
    generate_candidate,
    postprocess_region,
    validate_texture,
)


def _item_texture():
    request = AssetRequest(
        "credit_item", "item", "golden space credits", "item.generated", "credit_item",
    )
    return resolve_asset(
        request, namespace="alpha_test", minecraft_version="1.21.4",
    ).textures[0]


def _opaque_diffusion_image(width=512, height=512):
    image = Image.new("RGBA", (width, height))
    # Opaque, non-flat corner and gradient: the old exact-corner flood fill
    # cannot infer a common background color and leaves every alpha=255.
    pixels = [
        (120 + x * 70 // width, 40 + y * 60 // height,
         10 + (x + y) * 40 // (width + height), 255)
        for y in range(height) for x in range(width)
    ]
    image.putdata(pixels)
    return image


@pytest.mark.parametrize("silhouette", ["cube", "cylinder"])
def test_opaque_variegated_diffusion_sprite_gets_explicit_host_alpha(silhouette, tmp_path):
    texture = _item_texture()
    with _opaque_diffusion_image() as image:
        processed = postprocess_region(
            image, texture.resource_contract, (16, 16), silhouette=silhouette,
        )
    try:
        alpha = set(processed.getchannel("A").getdata())
        assert alpha == {0, 255}
        assert processed.getpixel((0, 0)) == (0, 0, 0, 0)
        assert processed.getpixel((8, 8))[3] == 255
        assert processed.info["mmm_alpha_matte"] == "host_explicit_" + silhouette
        path = tmp_path / "item.png"
        processed.save(path)
        assert validate_texture(path, texture.to_dict())["status"] == "PASS"
    finally:
        processed.close()


def test_unapproved_silhouette_does_not_erase_unknown_geometry(tmp_path):
    texture = _item_texture()
    with _opaque_diffusion_image() as image:
        processed = postprocess_region(
            image, texture.resource_contract, (16, 16), silhouette="complex spaceship",
        )
    try:
        path = tmp_path / "unknown.png"
        processed.save(path)
        with pytest.raises(ValueError, match="transparent alpha"):
            validate_texture(path, texture.to_dict())
        assert "mmm_alpha_matte" not in processed.info
    finally:
        processed.close()


def test_existing_model_alpha_is_never_replaced_by_geometric_host_mask():
    texture = _item_texture()
    source = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    source.paste((30, 90, 200, 255), (3, 3, 29, 29))
    try:
        processed = postprocess_region(
            source, texture.resource_contract, (16, 16), silhouette="cube",
        )
        try:
            assert processed.getpixel((2, 8))[3] == 255
            assert processed.getpixel((0, 0))[3] == 0
            assert "mmm_alpha_matte" not in processed.info
        finally:
            processed.close()
    finally:
        source.close()


def test_actual_candidate_generation_uses_segmenter_not_shape_mask(tmp_path):
    texture = _item_texture()
    calls = []

    def generator(**kwargs):
        calls.append(kwargs["seed"])
        with _opaque_diffusion_image(kwargs["width"], kwargs["height"]) as image:
            image.save(kwargs["output_path"], "PNG")

    output = tmp_path / "normalized.png"
    def fake_segmenter(image):
        from PIL import ImageDraw
        result = image.convert("RGBA")
        alpha = Image.new("L", result.size, 0)
        ImageDraw.Draw(alpha).ellipse((50, 40, 460, 475), fill=255)
        result.putalpha(alpha)
        alpha.close()
        return result

    evidence = generate_candidate(
        generator, texture.to_dict(),
        prompt="gold cube", directory=tmp_path / "regions",
        output=output, resolution=(512, 512), seed=11, silhouette="cube",
        segment_foreground_callback=fake_segmenter,
    )
    assert len(calls) == 1
    assert evidence["sources"][0]["alpha_matte"] == "rembg:birefnet-general"
    assert validate_texture(output, texture.to_dict())["status"] == "PASS"
