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


def test_opaque_flux_background_is_not_faked_into_alpha_by_geometry(tmp_path):
    texture = _item_texture()
    with _opaque_diffusion_image() as image:
        processed = postprocess_region(image, texture.resource_contract, (16, 16))
    try:
        path = tmp_path / "unsegmented.png"
        processed.save(path)
        with pytest.raises(ValueError, match="transparent alpha"):
            validate_texture(path, texture.to_dict())
    finally:
        processed.close()


def test_existing_model_alpha_is_preserved():
    texture = _item_texture()
    source = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    source.paste((30, 90, 200, 255), (3, 3, 29, 29))
    try:
        processed = postprocess_region(
            source, texture.resource_contract, (16, 16),
        )
        try:
            assert processed.getpixel((2, 8))[3] == 255
            assert processed.getpixel((0, 0))[3] == 0
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
        output=output, resolution=(512, 512), seed=11,
        segment_foreground_callback=fake_segmenter,
    )
    assert len(calls) == 1
    assert evidence["sources"][0]["alpha_matte"] == "rembg:birefnet-general-lite"
    assert validate_texture(output, texture.to_dict())["status"] == "PASS"


def test_all_regions_generated_then_flux_released_before_any_onnx(tmp_path, monkeypatch):
    from minecraft_mod_ai import resource_image_pipeline as pipeline
    texture = _item_texture()
    events = []

    # Two regions exercise the model-residency boundary without loading FLUX.
    monkeypatch.setattr(pipeline, "generation_regions", lambda _contract: [
        {"name": "first", "box": [0, 0, 16, 16]},
        {"name": "second", "box": [0, 0, 16, 16]},
    ])

    def generator(**kwargs):
        events.append("generate")
        with _opaque_diffusion_image(kwargs["width"], kwargs["height"]) as image:
            image.save(kwargs["output_path"], "PNG")

    def matting(image):
        from PIL import ImageDraw
        events.append("matte")
        result = image.convert("RGBA")
        with Image.new("L", result.size, 0) as alpha:
            ImageDraw.Draw(alpha).ellipse((60, 60, 440, 440), fill=255)
            result.putalpha(alpha)
        return result

    output = tmp_path / "candidate.png"
    pipeline.generate_candidate(
        generator, texture.to_dict(),
        prompt="two frame asset", directory=tmp_path / "regions",
        output=output, resolution=(512, 512), seed=5,
        before_segmentation=lambda: events.append("release"),
        segment_foreground_callback=matting,
    )
    assert events == ["generate", "generate", "release", "matte", "matte"]
    assert validate_texture(output, texture.to_dict())["status"] == "PASS"
