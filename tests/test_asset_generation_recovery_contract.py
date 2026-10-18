"""Regressions for FLUX/ONNX retry residency and native crystal-block detail."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from minecraft_mod_ai.complete_spec import AssetRequest
from minecraft_mod_ai.resource_contracts import resolve_asset
from minecraft_mod_ai.resource_image_pipeline import finalize_candidate_sources, validate_texture


def _block_texture():
    request = AssetRequest(
        asset_id="texture_block_crystal_block",
        kind="block",
        visual_description="crystal block with blue facets",
        render_kind="block.cube_all",
        subject_id="crystal_block",
    )
    return resolve_asset(
        request, namespace="example", minecraft_version="1.21.4",
    ).textures[0].to_dict()


def test_block_recovery_uses_only_real_high_resolution_source_detail(tmp_path):
    """Tiny genuine facets between nearest-neighbour samples survive recropping."""
    texture = _block_texture()
    source = tmp_path / "render.png"
    output = tmp_path / "block.png"
    with Image.new("RGB", (512, 512), (16, 28, 44)) as image:
        # Standard 512->16 nearest-neighbour sampling misses this 8px facet.
        image.paste((70, 170, 245), (251, 251, 259, 259))
        image.save(source)
    evidence = finalize_candidate_sources(
        texture,
        [({"name": "frame_0", "box": [0, 0, 16, 16]}, source, False)],
        output=output, resolution=(512, 512),
    )
    assert evidence["sources"][0]["native_detail_recovered"] is True
    assert validate_texture(output, texture)["status"] == "PASS"
    with Image.open(output) as image:
        colors = set(image.convert("RGB").getdata())
    assert colors <= {(16, 28, 44), (70, 170, 245)}
    assert len(colors) == 2


def test_monochrome_source_is_never_painted_into_a_fake_texture(tmp_path):
    texture = _block_texture()
    source = tmp_path / "uniform.png"
    output = tmp_path / "block.png"
    with Image.new("RGB", (512, 512), (16, 28, 44)) as image:
        image.save(source)
    with pytest.raises(ValueError, match="VISUAL_TEXTURE_DEGENERATE"):
        finalize_candidate_sources(
            texture,
            [({"name": "frame_0", "box": [0, 0, 16, 16]}, source, False)],
            output=output, resolution=(512, 512),
        )


def test_removed_image_generation_rejects_before_loading_any_backend(tmp_path):
    from minecraft_mod_ai import resource_asset_production as assets

    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"removed generator accessed: {name}")

    proposal = SimpleNamespace(assets=(object(),))
    with pytest.raises(ValueError, match="AI_TEXTURE_GENERATION_REMOVED"):
        assets.generate_assets(NeverModel(), proposal, tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())
