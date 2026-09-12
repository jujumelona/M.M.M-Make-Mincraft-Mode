from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.complete_spec import (
    AssetRequest,
    ProductionModule,
    _asset_from_dict,
)
from minecraft_mod_ai.resource_contracts import derive_module_asset_specs, resolve_asset

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_asset_input_migrates_only_at_deserialization_boundary() -> None:
    asset = _asset_from_dict(
        {
            "asset_id": "texture_item_blade",
            "kind": "item",
            "prompt": "dark steel blade",
            "target_path": "src/main/resources/assets/demo/textures/item/blade.png",
            "width": 32,
            "height": 32,
        }
    )
    assert asset.visual_description == "dark steel blade"
    assert asset.subject_id == "blade"
    assert asset.render_kind == "item.generated"
    assert asset.requested_width == 32
    assert asset.requested_height == 32

    params = inspect.signature(AssetRequest).parameters
    assert not {"prompt", "target_path", "width", "height"} & set(params)
    assert not hasattr(asset, "prompt")
    assert not hasattr(asset, "width")
    assert not hasattr(asset, "height")

    with pytest.raises(TypeError):
        AssetRequest(
            asset_id="texture_item_blade",
            kind="item",
            visual_description="dark steel blade",
            render_kind="item.generated",
            subject_id="blade",
            prompt="legacy prompt must not enter the canonical contract",
        )


def test_module_assets_are_host_derived_and_owned() -> None:
    module = ProductionModule(module_id="moon_blade", kind="weapon", config={"display_name_en": "Moon Blade", "material": "steel"})
    row = derive_module_asset_specs((module,))[0]
    assert row["asset_id"] == "texture_item_moon_blade"
    assert row["render_kind"] == "item.handheld"
    assert row["owner_module_id"] == "moon_blade"


def test_item_serialization_follows_host_client_item_binding() -> None:
    asset = AssetRequest("texture_item_blade", "item", visual_description="dark steel blade", render_kind="item.generated", subject_id="blade")
    old = resolve_asset(asset, namespace="demo", minecraft_version="1.21.3")
    context = SimpleNamespace(facts={"leaf_bindings": {leaf: {"implementation": {"extra_templates": ["minecraft/resource/item/client_item"]}} for leaf in ("minecraft/item/model", "minecraft/block/model")}}, admit_template=lambda template: None)
    new = resolve_asset(asset, namespace="demo", minecraft_version="1.21.4", version_context=context)
    assert "src/main/resources/assets/demo/items/blade.json" not in {d.target_path for d in old.documents}
    assert "src/main/resources/assets/demo/items/blade.json" in {d.target_path for d in new.documents}


def test_uv_and_gui_require_host_geometry_instead_of_generic_defaults() -> None:
    entity = AssetRequest(
        "texture_entity_guardian",
        "entity",
        visual_description="stone guardian",
        render_kind="entity.fixed_uv",
        subject_id="guardian",
    )
    gui = AssetRequest(
        "texture_gui_console",
        "gui",
        visual_description="arcane console",
        render_kind="gui.sprite",
        subject_id="console",
    )

    for asset in (entity, gui):
        with pytest.raises(ValueError, match="HOST"):
            resolve_asset(asset, namespace="demo", minecraft_version="1.21.4")


def test_content_design_graph_only_emits_semantic_asset_requests() -> None:
    source = (ROOT / "minecraft_mod_ai/content_design_graph.py").read_text(encoding="utf-8")
    for obsolete in (
        "asset/item_sprite",
        "asset/block_tile",
        "asset/entity_texture",
        "asset/gui_panel",
        "target_path=f\"assets/{mod_id}/textures/",
        "prompt=prompt_text",
        "width=w",
        "height=h",
    ):
        assert obsolete not in source
    assert "visual_description=visual_desc" in source
    assert "owner_module_id=eid" in source
    assert "subject_id=eid" in source


def test_backend_literals_are_not_owned_by_resource_orchestrator() -> None:
    production = (ROOT / "minecraft_mod_ai/resource_asset_production.py").read_text(encoding="utf-8")
    backend = (ROOT / "minecraft_mod_ai/model_adapters/image_diffusion.py").read_text(encoding="utf-8")
    assert "black-forest-labs/FLUX.2-klein-9B" not in production
    assert "PIXELART-REDMOND" not in production
    assert "generate_fixed_template_text" not in production
    assert "ImageGenerationConfig" in backend


def test_no_placeholder_texture_writer_in_active_paths() -> None:
    assert "make_texture_png" not in (ROOT / "minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")
    assert "make_texture_png" not in (ROOT / "minecraft_mod_ai/extended_content_generator.py").read_text(encoding="utf-8")


def test_asset_prompt_fragment_consumes_resolved_authorities() -> None:
    from minecraft_mod_ai.task_template_catalog import load_template
    template = load_template("asset/item_sprite")
    assert set(template["requires"]) == {"resolved_resource_contract", "resolved_generation_profile", "visual_spec"}
    assert "PixArFK" not in template["render"]["body"]
    for relative in ("asset/block_tile.yaml", "asset/entity_texture.yaml", "asset/gui_panel.yaml"):
        assert not (ROOT / "minecraft_mod_ai/templates" / relative).exists()

    validation = (ROOT / "minecraft_mod_ai/template_contract_validation.py").read_text(encoding="utf-8")
    for identifier in ("asset/block_tile", "asset/entity_texture", "asset/gui_panel"):
        assert identifier not in validation


def test_resume_cache_binds_only_to_canonical_asset_producer() -> None:
    from minecraft_mod_ai import (
        complete_orchestrator_services,
        resource_asset_production,
    )

    assert getattr(
        resource_asset_production.generate_assets,
        "_mmm_resumable_image_sources",
        False,
    )
    assert not hasattr(
        complete_orchestrator_services,
        "_generate_single_asset_source",
    )
    assert not hasattr(
        complete_orchestrator_services,
        "_generate_tiled_asset_source",
    )
