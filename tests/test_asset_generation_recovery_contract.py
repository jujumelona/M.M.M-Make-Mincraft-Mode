"""Regressions for FLUX/ONNX retry residency and native crystal-block detail."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
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


def test_all_retry_images_are_generated_before_single_alpha_handoff(monkeypatch, tmp_path):
    from minecraft_mod_ai import resource_asset_production as assets
    from minecraft_mod_ai import resource_image_pipeline as pipeline
    from minecraft_mod_ai import resource_alpha_segmentation as alpha
    from minecraft_mod_ai.model_adapters import image_diffusion
    from minecraft_mod_ai.model_adapters.image_diffusion import ImageGenerationConfig

    events = []
    prompts = {}
    textures = [
        {
            "role": role, "prompt": role,
            "target_path": f"src/main/resources/assets/example/textures/item/{role}.png",
            "resource_contract": {"animation": {"mcmeta_required": False}},
            "alpha_policy": "cutout", "topology": "cutout_sprite",
            "width": 16, "height": 16,
        }
        for role in ("first", "second")
    ]
    rows = [{"asset_id": "asset", "container": "mod",
             "render_kind": "item.generated", "textures": textures, "documents": []}]

    monkeypatch.setattr(assets, "_preflight", lambda _: None)
    monkeypatch.setattr(assets, "_validated_plan", lambda *_: {"assets": rows})
    monkeypatch.setattr(assets, "_write_documents", lambda *_: [])
    monkeypatch.setattr(assets, "_validate_reference_closure",
                        lambda *_, **__: {"status": "PASS", "checked_reference_count": 0})
    monkeypatch.setattr(assets, "_validate_container_layout",
                        lambda *_, **__: ({"mod_resources": {"status": "PASS"}}, ""))
    monkeypatch.setattr(pipeline, "validate_model_consumers", lambda *_: None)
    monkeypatch.setattr(pipeline, "validate_texture",
                        lambda *_, **__: {"status": "PASS"})
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 12 * 1024**3)
    monkeypatch.setattr(alpha, "prepare_foreground_model_isolated", lambda: None)
    monkeypatch.setattr(
        ImageGenerationConfig, "from_adapter_config",
        classmethod(lambda cls, _: SimpleNamespace(
            candidate_count=3, preferred_generation_resolution=(256, 256),
        )),
    )
    monkeypatch.setattr(
        image_diffusion, "release_image_pipeline_for_segmentation",
        lambda: events.append("release"),
    )

    def prepare(_generator, texture, **kwargs):
        role = texture["role"]
        attempt = int(kwargs["directory"].name.split("-")[-1])
        events.append(f"prepare:{role}:{attempt}")
        prompts[(role, attempt)] = kwargs["prompt"]
        return [({"name": role, "box": [0, 0, 16, 16]},
                 kwargs["directory"] / "source.png", True)]

    def finish(texture, pending, *, output, resolution):
        role = texture["role"]
        attempt = int(pending[0][1].parent.name.split("-")[-1])
        events.append(f"finish:{role}:{attempt}")
        if role == "first" and attempt == 0:
            raise ValueError("VISUAL_TEXTURE_DEGENERATE")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"accepted-real-image-fixture")
        return {"sources": [{"alpha_matte": "rembg:u2netp"}]}

    monkeypatch.setattr(pipeline, "prepare_candidate_sources", prepare)
    monkeypatch.setattr(pipeline, "finalize_candidate_sources", finish)

    class Router:
        profile = "test"
        registry = SimpleNamespace(role=lambda *_: object())

        @contextmanager
        def image_generation_session(self, role):
            yield self

    proposal = SimpleNamespace(
        assets=(object(),),
        base_proposal=SimpleNamespace(spec=SimpleNamespace(mod_id="example")),
    )
    receipt = assets.generate_assets(
        Router(), proposal, tmp_path / "project", tmp_path / "run",
        package_resource_pack=False,
    )
    assert events == [
        "prepare:first:0", "prepare:second:0",
        "prepare:first:1", "prepare:second:1",
        "prepare:first:2", "prepare:second:2",
        "release",
        "finish:first:0", "finish:first:1", "finish:second:0",
    ]
    assert receipt["count"] == 2
    assert [r["attempted_candidate_count"] for r in receipt["assets"]] == [2, 1]
    assert prompts[("first", 0)] != prompts[("first", 1)]
    assert len(receipt["assets"][0]["rejected_candidates"]) == 1
