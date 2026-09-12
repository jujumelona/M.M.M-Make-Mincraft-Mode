from dataclasses import replace
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.complete_spec import AssetRequest
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.resource_contracts import resolve_asset
from minecraft_mod_ai.resource_prompt_compiler import image_profile_fingerprint


def request(kind="item", render="item.generated"):
    return AssetRequest("blade", kind, "obsidian blade", render, "blade")


def context(binding):
    return SimpleNamespace(
        facts={
            "resource_asset_bindings": {"blade": binding},
            "leaf_bindings": {
                leaf: {"implementation": {"extra_templates": []}}
                for leaf in ("minecraft/item/model", "minecraft/block/model")
            },
        },
        admit_template=lambda template: None,
    )


def test_full_contract_separates_native_output_and_generation():
    texture = resolve_asset(
        request(), namespace="demo", minecraft_version="1.21.4"
    ).textures[0]
    contract = texture.to_dict()["resource_contract"]
    assert contract["geometry"]["canonical_width"] == 16
    assert contract["model"]["family"] == "item.generated"
    assert contract["generation_profile_ref"] == "image_generator"
    assert "generation_resolution" not in contract["geometry"]


@pytest.mark.parametrize(
    "kind,render", [("entity", "entity.fixed_uv"), ("gui", "gui.sprite")]
)
def test_structured_layout_requires_host_binding(kind, render):
    with pytest.raises(ValueError, match="HOST"):
        resolve_asset(
            request(kind, render), namespace="demo", minecraft_version="1.21.4"
        )


def test_visual_spec_rejects_technical_authority():
    from minecraft_mod_ai.resource_visual_spec import VisualSpec

    with pytest.raises(ValueError, match="visual"):
        VisualSpec.from_dict({"role": "weapon", "checkpoint": "injected"})
    spec = VisualSpec.from_dict(
        {"role": "weapon", "materials": ["obsidian"], "palette": {"accent": "orange"}}
    )
    assert "obsidian" in spec.prompt_fragment()


def test_resolution_changes_invalidate_approved_profile():
    config = ModelRegistry().role("t4_local", "image_generator")
    changed = replace(
        config,
        extra={
            **config.extra,
            "preferred_generation_resolution": {"width": 512, "height": 512},
        },
    )
    assert image_profile_fingerprint(config) != image_profile_fingerprint(changed)


def test_animation_expands_sheet_and_mcmeta_from_host():
    binding = {"animation": {"frame_count": 3, "frametime": 2}}
    resolved = resolve_asset(
        request(),
        namespace="demo",
        minecraft_version="1.21.4",
        version_context=context(binding),
    )
    assert (resolved.textures[0].width, resolved.textures[0].height) == (16, 48)
    metadata = next(
        d for d in resolved.documents if d.target_path.endswith(".png.mcmeta")
    )
    assert metadata.payload["animation"]["frames"] == [0, 1, 2]


def test_ai_dimensions_cannot_override_host_geometry():
    with pytest.raises(ValueError, match="HOST"):
        resolve_asset(
            replace(request(), requested_width=64, requested_height=64),
            namespace="demo",
            minecraft_version="1.21.4",
        )


def test_postprocess_and_validation_use_actual_pixels(tmp_path):
    from PIL import Image

    from minecraft_mod_ai.resource_image_pipeline import (
        postprocess_region,
        validate_texture,
    )

    texture = resolve_asset(
        request(), namespace="demo", minecraft_version="1.21.4"
    ).textures[0]
    source = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    source.paste((20, 30, 40, 255), (128, 128, 384, 384))
    output = postprocess_region(source, texture.resource_contract, (16, 16))
    path = tmp_path / "blade.png"
    output.save(path)
    assert output.getpixel((0, 0))[3] == 0
    assert output.getpixel((8, 8))[3] == 255
    validate_texture(path, texture.to_dict())
    Image.new("RGBA", (16, 16), (20, 30, 40, 255)).save(path)
    with pytest.raises(ValueError, match="alpha"):
        validate_texture(path, texture.to_dict())


def test_uv_regions_are_generated_separately_and_composed(tmp_path):
    from PIL import Image

    from minecraft_mod_ai.resource_image_pipeline import (
        generate_candidate,
        validate_texture,
    )

    binding = {
        "geometry": {"width": 32, "height": 16},
        "model_binding": {
            "path": "renderer.json",
            "sha256": "sha256:" + "a" * 64,
            "texture_reference": "demo:textures/entity/blade.png",
        },
        "uv_schema": {
            "id": "guardian-v1",
            "lora_compatible": True,
            "regions": [
                {"name": "head", "box": [0, 0, 8, 8]},
                {"name": "body", "box": [16, 0, 16, 16]},
            ],
        },
    }
    texture = resolve_asset(
        request("entity", "entity.fixed_uv"),
        namespace="demo",
        minecraft_version="1.21.4",
        version_context=context(binding),
    ).textures[0]
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (512, 512), (len(calls) * 80, 10, 20, 255)).save(
            kwargs["output_path"]
        )

    output = tmp_path / "atlas.png"
    generate_candidate(
        generate,
        texture.to_dict(),
        prompt="stone",
        directory=tmp_path,
        output=output,
        resolution=(512, 512),
        fallback=None,
        seed=1,
    )
    assert len(calls) == 2
    with Image.open(output) as image:
        assert image.getpixel((0, 0)) == (80, 10, 20, 255)
        assert image.getpixel((16, 0)) == (160, 10, 20, 255)
        assert image.getpixel((10, 0))[3] == 0
    validate_texture(output, texture.to_dict())


def test_gui_protected_regions_survive_generated_decoration(tmp_path):
    from PIL import Image

    from minecraft_mod_ai.resource_image_pipeline import (
        generate_candidate,
        validate_texture,
    )

    binding = {
        "geometry": {"width": 32, "height": 32},
        "model_binding": {
            "path": "screen.json",
            "sha256": "sha256:" + "a" * 64,
            "texture_reference": "demo:textures/gui/blade.png",
        },
        "gui": {
            "protected_regions": [
                {"name": "slot", "box": [4, 4, 8, 8], "rgba": [10, 20, 30, 255]}
            ],
            "generated_regions": [{"name": "background", "box": [0, 0, 32, 32]}],
        },
    }
    texture = resolve_asset(
        request("gui", "gui.sprite"),
        namespace="demo",
        minecraft_version="1.21.4",
        version_context=context(binding),
    ).textures[0]

    def generate(**kwargs):
        Image.new("RGBA", (512, 512), (200, 100, 50, 255)).save(kwargs["output_path"])

    output = tmp_path / "gui.png"
    generate_candidate(
        generate,
        texture.to_dict(),
        prompt="brass",
        directory=tmp_path,
        output=output,
        resolution=(512, 512),
        fallback=None,
        seed=1,
    )
    validate_texture(output, texture.to_dict())
    with Image.open(output) as original:
        corrupted = original.convert("RGBA")
    corrupted.putpixel((4, 4), (200, 100, 50, 255))
    corrupted.save(output)
    with pytest.raises(ValueError, match="protected"):
        validate_texture(output, texture.to_dict())


def test_tile_edges_are_repaired_and_enforced(tmp_path):
    from PIL import Image

    from minecraft_mod_ai.resource_image_pipeline import (
        postprocess_region,
        validate_texture,
    )

    texture = resolve_asset(
        request("block", "block.cube_all"), namespace="demo", minecraft_version="1.21.4"
    ).textures[0]
    source = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
    source.paste((200, 100, 50, 255), (0, 0, 1, 16))
    result = postprocess_region(source, texture.resource_contract, (16, 16))
    output = tmp_path / "tile.png"
    result.save(output)
    validate_texture(output, texture.to_dict())
    result.putpixel((0, 5), (1, 2, 3, 255))
    result.save(output)
    with pytest.raises(ValueError, match="tile"):
        validate_texture(output, texture.to_dict())


def runtime():
    from contextlib import nullcontext

    from PIL import Image

    from minecraft_mod_ai.resource_asset_production import _plan_row

    config = ModelRegistry().role("t4_local", "image_generator")
    config = replace(config, extra={**config.extra, "candidate_count": 1})
    calls = []

    def generate(role, **kwargs):
        calls.append(kwargs)
        image = Image.new("RGBA", (kwargs["width"], kwargs["height"]), (0, 0, 0, 0))
        image.paste((20, 40, 80, 255), (128, 128, 384, 384))
        image.save(kwargs["output_path"])
        return kwargs["output_path"]

    router = SimpleNamespace(
        registry=SimpleNamespace(role=lambda *args: config),
        profile="fixture",
        image_generation_session=lambda *args: nullcontext(),
        generate_image=generate,
    )
    proposal = SimpleNamespace(
        assets=(request(),),
        modules=(),
        game_design={},
        base_proposal=SimpleNamespace(
            spec=SimpleNamespace(
                mod_id="demo",
                platform=SimpleNamespace(
                    minecraft_version="1.21.4", host_facts_json=""
                ),
            )
        ),
    )
    proposal.validate = lambda: request().validate()
    proposal.game_design["_asset_generation_plan"] = {
        "schema_version": "mmm/resource-asset-generation-plan-v3",
        "image_profile_sha256": image_profile_fingerprint(config),
        "assets": [_plan_row(router, proposal, request())],
    }
    return router, proposal, calls


def test_production_consumes_registry_and_checks_real_resource_set(tmp_path):
    from PIL import Image

    from minecraft_mod_ai.resource_asset_production import generate_assets

    router, proposal, calls = runtime()
    result = generate_assets(router, proposal, tmp_path / "project", tmp_path / "run")
    assert calls[0]["width"] == 1024
    assert result["resource_contract_validation"]["status"] == "PASS"
    assert result["resource_graph_validation"]["checked_reference_count"] == 1
    with Image.open(result["assets"][0]["target"]) as image:
        assert image.size == (16, 16)


def test_tampered_manifest_is_rejected_before_backend(tmp_path):
    from minecraft_mod_ai.resource_asset_production import (
        AssetProductionError,
        generate_assets,
    )

    router, proposal, calls = runtime()
    proposal.game_design["_asset_generation_plan"]["assets"][0]["textures"] = []
    with pytest.raises(AssetProductionError, match="manifest"):
        generate_assets(router, proposal, tmp_path, tmp_path / "run")
    assert calls == []


def test_request_cannot_select_model_family_or_crop_cardinality():
    with pytest.raises(ValueError, match="HOST"):
        resolve_asset(
            request("block", "block.orientable"),
            namespace="demo",
            minecraft_version="1.21.4",
        )
    with pytest.raises(ValueError, match="cardinality"):
        resolve_asset(
            replace(request("block", "block.crop"), variant_count=8),
            namespace="demo",
            minecraft_version="1.21.4",
            version_context=context({"render_kind": "block.crop", "variant_count": 4}),
        )


def test_stale_animation_metadata_is_not_accepted_as_static(tmp_path):
    from PIL import Image

    from minecraft_mod_ai.resource_image_pipeline import validate_texture

    texture = resolve_asset(
        request("block", "block.cube_all"), namespace="demo", minecraft_version="1.21.4"
    ).textures[0]
    path = tmp_path / "block.png"
    Image.new("RGBA", (16, 16), (20, 40, 80, 255)).save(path)
    (tmp_path / "block.png.mcmeta").write_text('{"animation": {}}')
    with pytest.raises(ValueError, match="Static"):
        validate_texture(path, texture.to_dict(), check_metadata=True)


def test_consumer_binding_verifies_actual_source_and_exact_reference(tmp_path):
    import hashlib

    from minecraft_mod_ai.resource_image_pipeline import validate_model_consumers

    data = b'{"texture":"demo:textures/entity/blade.png"}'
    (tmp_path / "renderer.json").write_bytes(data)
    binding = {
        "geometry": {"width": 16, "height": 16},
        "model_binding": {
            "path": "renderer.json",
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "texture_reference": "demo:textures/entity/blade.png",
        },
        "uv_schema": {
            "id": "test",
            "regions": [{"name": "body", "box": [0, 0, 16, 16]}],
        },
    }
    texture = resolve_asset(
        request("entity", "entity.fixed_uv"),
        namespace="demo",
        minecraft_version="1.21.4",
        version_context=context(binding),
    ).textures[0]
    validate_model_consumers(tmp_path, [texture.to_dict()])
    (tmp_path / "renderer.json").write_text("modified")
    with pytest.raises(ValueError, match="hash"):
        validate_model_consumers(tmp_path, [texture.to_dict()])


def test_structured_semantics_deserialize_without_freeform_prompt():
    from minecraft_mod_ai.complete_spec import _asset_from_dict

    asset = _asset_from_dict(
        {
            "asset_id": "blade",
            "kind": "item",
            "render_kind": "item.generated",
            "subject_id": "blade",
            "visual_spec": {"role": "weapon", "materials": ["obsidian"]},
        }
    )
    asset.validate()
    assert asset.visual_spec["materials"] == ["obsidian"]


def test_animation_generation_writes_distinct_frames_and_validates_sidecar(tmp_path):
    import json

    from PIL import Image

    from minecraft_mod_ai.resource_image_pipeline import (
        generate_candidate,
        validate_texture,
    )

    resolved = resolve_asset(request("block", "block.cube_all"), namespace="demo", minecraft_version="1.21.4",
                             version_context=context({"animation": {"frame_count": 2, "frametime": 3}}))
    texture = resolved.textures[0]
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        Image.new("RGBA", (512, 512), (40 * len(calls), 20, 30, 255)).save(kwargs["output_path"])
    output = tmp_path / "animated.png"
    generate_candidate(generate, texture.to_dict(), prompt="stone", directory=tmp_path, output=output,
                       resolution=(512, 512), fallback=None, seed=1)
    assert len(calls) == 2 and calls[0]["seed"] != calls[1]["seed"]
    with Image.open(output) as sheet:
        assert sheet.size == (16, 32)
        assert sheet.getpixel((8, 8)) != sheet.getpixel((8, 24))
    metadata = next(d.payload for d in resolved.documents if d.target_path.endswith(".mcmeta"))
    (tmp_path / "animated.png.mcmeta").write_text(json.dumps(metadata))
    validate_texture(output, texture.to_dict(), check_metadata=True)
    (tmp_path / "animated.png.mcmeta").write_text('{"animation": {}}')
    with pytest.raises(ValueError, match="mcmeta"):
        validate_texture(output, texture.to_dict(), check_metadata=True)


def test_only_memory_failure_uses_registry_fallback(tmp_path):
    from PIL import Image

    from minecraft_mod_ai.model_adapters.base import ModelBackendError
    from minecraft_mod_ai.resource_image_pipeline import generate_candidate

    texture = resolve_asset(request("block", "block.cube_all"), namespace="demo", minecraft_version="1.21.4").textures[0]
    calls = []
    def generate(**kwargs):
        calls.append(kwargs["width"])
        if kwargs["width"] == 1024:
            raise ModelBackendError(role="image_generator", model_id="fixture", cause="CUDA out of memory")
        Image.new("RGBA", (512, 512), (20, 40, 80, 255)).save(kwargs["output_path"])
    evidence = generate_candidate(generate, texture.to_dict(), prompt="stone", directory=tmp_path,
                                  output=tmp_path / "result.png", resolution=(1024, 1024), fallback=(512, 512), seed=1)
    assert calls == [1024, 512]
    assert evidence["sources"][0]["resolution"] == [512, 512]
    def broken(**kwargs):
        raise ModelBackendError(role="image_generator", model_id="fixture", cause="authentication rejected")
    with pytest.raises(ModelBackendError, match="authentication"):
        generate_candidate(broken, texture.to_dict(), prompt="stone", directory=tmp_path,
                           output=tmp_path / "result.png", resolution=(1024, 1024), fallback=(512, 512), seed=1)


def test_host_uv_requires_explicit_lora_compatibility():
    from minecraft_mod_ai.resource_prompt_compiler import compile_texture_prompt

    binding = {"geometry": {"width": 16, "height": 16},
               "model_binding": {"path": "renderer.json", "sha256": "sha256:" + "a" * 64, "texture_reference": "demo:textures/entity/blade.png"},
               "uv_schema": {"id": "test", "regions": [{"name": "body", "box": [0, 0, 16, 16]}]}}
    texture = resolve_asset(request("entity", "entity.fixed_uv"), namespace="demo", minecraft_version="1.21.4", version_context=context(binding)).textures[0]
    with pytest.raises(ValueError, match="LoRA compatibility"):
        compile_texture_prompt(visual_description="stone", texture=texture, image_config=ModelRegistry().role("t4_local", "image_generator"))
