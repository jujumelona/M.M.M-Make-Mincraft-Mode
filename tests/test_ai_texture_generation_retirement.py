"""No FLUX / standalone resource-pack branch is reachable from new M.M.M plans."""
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.capability_plugins import buildable_plugin_ids
from minecraft_mod_ai.complete_spec import AssetRequest, SpecValidationError
from minecraft_mod_ai.extended_content_generator import _block_resources, _item_resources
from minecraft_mod_ai.resource_asset_production import (
    attach_generation_plan, generate_assets,
)
from minecraft_mod_ai.visual_generation_policy import (
    AI_TEXTURE_GENERATION_ENABLED,
    STANDALONE_RESOURCE_PACK_GENERATION_ENABLED,
    reject_generated_assets,
)


def test_no_image_backend_or_standalone_resourcepack_capability():
    assert not AI_TEXTURE_GENERATION_ENABLED
    assert not STANDALONE_RESOURCE_PACK_GENERATION_ENABLED
    assert "asset-generation" not in buildable_plugin_ids()
    with pytest.raises(ValueError, match="AI_TEXTURE_GENERATION_REMOVED"):
        reject_generated_assets([object()])


def test_saved_standalone_resource_pack_is_rejected():
    row = AssetRequest(
        asset_id="pack_texture", kind="item", visual_description="iron ore icon",
        render_kind="item.generated", subject_id="ore",
        container="resource_pack",
    )
    with pytest.raises(SpecValidationError, match="STANDALONE_RESOURCE_PACK_GENERATION_REMOVED"):
        row.validate()


def test_image_plan_never_initializes_router_or_flux():
    proposal = SimpleNamespace(assets=[object()])
    with pytest.raises(ValueError, match="AI_TEXTURE_GENERATION_REMOVED"):
        attach_generation_plan(None, proposal)


def test_direct_asset_generation_is_disabled_before_backend_setup(tmp_path):
    proposal = SimpleNamespace(assets=[object()])
    with pytest.raises(ValueError, match="AI_TEXTURE_GENERATION_REMOVED"):
        generate_assets(None, proposal, tmp_path, tmp_path)


def test_extended_content_model_json_uses_vanilla_texture_fallbacks():
    item = _item_resources("modid", "special_sword", "weapon", {})
    item_model = json.loads(item["src/main/resources/assets/modid/models/item/special_sword.json"])
    assert item_model["textures"]["layer0"] == "minecraft:item/iron_sword"
    block = _block_resources("modid", "new_ore", "block", {})
    block_model = json.loads(block["src/main/resources/assets/modid/models/block/new_ore.json"])
    assert block_model["textures"]["all"] == "minecraft:block/stone"
    crop = _block_resources("modid", "example_crop", "crop", {})
    crop_model = json.loads(crop["src/main/resources/assets/modid/models/block/example_crop_stage0.json"])
    assert crop_model["textures"]["crop"] == "minecraft:block/wheat_stage0"


def test_colab_installer_does_not_request_image_stack():
    from minecraft_mod_ai import visual_generation_policy as visual
    root = Path(visual.__file__).resolve().parents[1]
    config = (root / "requirements-colab.txt").read_text(encoding="utf-8")
    assert ",image" not in config
    installer = (root / "tools/colab_runtime_setup.py").read_text(encoding="utf-8")
    assert '.[ui,local-model,rag,image' not in installer
    assert 'from diffusers import Flux2KleinPipeline' not in installer


def test_default_proposal_constructor_never_derives_image_requests():
    import inspect
    from minecraft_mod_ai.complete_spec import complete_proposal_from_parts
    source = inspect.getsource(complete_proposal_from_parts)
    assert "derive_module_asset_specs(" not in source
    assert "reject_generated_assets(assets)" in source
