from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import pytest

from minecraft_mod_ai.complete_spec import AssetRequest, ProductionModule
from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai import resource_asset_production as assets


def _asset() -> AssetRequest:
    return AssetRequest(
        asset_id="diamond_sword",
        kind="item",
        prompt="blue lightsaber replacement",
        target_path="assets/minecraft/textures/item/diamond_sword.png",
        width=16,
        height=16,
    )


def test_image_backend_is_fixed_to_klein9b_q4_pixelart_lora() -> None:
    assert assets.FLUX_MODEL_ID == "black-forest-labs/FLUX.2-klein-9B"
    assert assets.QUANTIZATION == "bnb_4bit_nf4"
    assert assets.PIXEL_LORA_ID == "artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B"
    assert assets.PIXEL_LORA_WEIGHT == "[FLUX.2.Klein]PixelArt_Redmond.safetensors"

    config = ModelRegistry().role("t4_local", "image_generator")
    assert config.model_id == assets.FLUX_MODEL_ID
    assert config.quantization == assets.QUANTIZATION
    assert config.extra["lora_model_id"] == assets.PIXEL_LORA_ID
    assert config.extra["lora_weight_name"] == assets.PIXEL_LORA_WEIGHT
    assert config.extra["lora_trigger"] == assets.PIXEL_LORA_TRIGGER


def test_backend_config_refuses_lora_or_model_drift() -> None:
    good = SimpleNamespace(
        model_id=assets.FLUX_MODEL_ID,
        quantization=assets.QUANTIZATION,
        extra={
            "lora_model_id": assets.PIXEL_LORA_ID,
            "lora_weight_name": assets.PIXEL_LORA_WEIGHT,
            "lora_trigger": assets.PIXEL_LORA_TRIGGER,
            "lora_scale": 1.0,
        },
    )
    assets._require_fixed_backend_config(good)
    bad = SimpleNamespace(
        model_id=assets.FLUX_MODEL_ID,
        quantization=assets.QUANTIZATION,
        extra={**good.extra, "lora_model_id": "wrong/lora"},
    )
    with pytest.raises(ModelConfigurationError):
        assets._require_fixed_backend_config(bad)


def test_persisted_plan_requires_four_real_prompts_and_exact_fixed_lora() -> None:
    request = _asset()
    prompts = [
        f"{assets.PIXEL_LORA_TRIGGER}. variant {index} blue lightsaber item, transparent background"
        for index in range(4)
    ]
    plan = {
        "schema_version": assets.PLAN_SCHEMA,
        "model_id": assets.FLUX_MODEL_ID,
        "quantization": assets.QUANTIZATION,
        "lora_model_id": assets.PIXEL_LORA_ID,
        "lora_weight_name": assets.PIXEL_LORA_WEIGHT,
        "lora_trigger": assets.PIXEL_LORA_TRIGGER,
        "prompt_candidates_per_asset": 4,
        "assets": [{
            "asset_id": request.asset_id,
            "kind": request.kind,
            "target_path": request.target_path,
            "width": 16,
            "height": 16,
            "prompts": prompts,
        }],
    }
    assert assets._valid_plan(plan, (request,))
    plan["assets"][0]["prompts"] = prompts[:3]
    assert not assets._valid_plan(plan, (request,))


def test_pixel_postprocess_uses_exact_rgba_size(tmp_path) -> None:
    source = tmp_path / "candidate.png"
    Image.new("RGBA", (256, 256), (10, 20, 30, 255)).save(source)
    result = assets._prepare_candidate(_asset(), 0, "prompt", source)
    with Image.open(result["normalized_path"]) as image:
        image.load()
        assert image.size == (16, 16)
        assert image.mode == "RGBA"


def test_capabilities_are_assigned_once_to_matching_modules() -> None:
    modules = (
        ProductionModule(module_id="trade_engine", kind="economy"),
        ProductionModule(module_id="shop_ui", kind="gui"),
    )
    decisions = (
        {"capability": "trade.transaction", "mode": "fresh"},
        {"capability": "ui.shop_menu", "mode": "fresh"},
    )
    ownership = assets._assign_capability_owners(modules, decisions)
    flat = [item["capability"] for rows in ownership.values() for item in rows]
    assert sorted(flat) == ["trade.transaction", "ui.shop_menu"]
    assert sum(item["capability"] == "trade.transaction" for rows in ownership.values() for item in rows) == 1
    assert sum(item["capability"] == "ui.shop_menu" for rows in ownership.values() for item in rows) == 1
