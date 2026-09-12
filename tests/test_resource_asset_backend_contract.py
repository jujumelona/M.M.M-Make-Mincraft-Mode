from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import resource_asset_production as assets
from minecraft_mod_ai.complete_spec import AssetRequest, ProductionModule
from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.model_adapters.image_diffusion import ImageGenerationConfig
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.resource_prompt_compiler import image_profile_fingerprint


def _asset() -> AssetRequest:
    return AssetRequest(
        asset_id="diamond_sword",
        kind="item",
        visual_description="blue lightsaber replacement",
        render_kind="item.handheld",
        subject_id="diamond_sword",
        container="resource_pack",
        requested_width=16,
        requested_height=16,
    )


def test_image_backend_is_registry_owned_klein9b_q4_pixelart_lora() -> None:
    config = ModelRegistry().role("t4_local", "image_generator")
    profile = ImageGenerationConfig.from_adapter_config(config)

    assert config.adapter == "image_diffusion"
    assert profile.model_id == "black-forest-labs/FLUX.2-klein-9B"
    assert profile.quantization == "bnb_4bit_nf4"
    assert profile.lora_model_id == "artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B"
    assert profile.lora_weight_name == "[FLUX.2.Klein]PixelArt_Redmond.safetensors"
    assert profile.lora_trigger == "Pixel Art, PixArFK"
    assert profile.candidate_count == 4
    assert profile.preferred_generation_resolution == (1024, 1024)
    assert profile.fallback_generation_resolution == (512, 512)


def test_image_profile_rejects_partial_lora_or_invalid_generation_geometry() -> None:
    base = SimpleNamespace(
        model_id="black-forest-labs/FLUX.2-klein-9B",
        quantization="bnb_4bit_nf4",
        torch_dtype="float16",
        cpu_offload=True,
        extra={
            "lora_model_id": "artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B",
            "lora_weight_name": "[FLUX.2.Klein]PixelArt_Redmond.safetensors",
            "lora_trigger": "Pixel Art, PixArFK",
            "candidate_count": 4,
            "preferred_generation_resolution": {"width": 1024, "height": 1024},
            "fallback_generation_resolution": {"width": 512, "height": 512},
            "lora_allowed_layouts": ["isolated_sprite"],
            "prompt_requirements": ["readable silhouette at native pixel grid"],
        },
    )
    ImageGenerationConfig.from_adapter_config(base)

    missing_weight = SimpleNamespace(
        **{
            **base.__dict__,
            "extra": {**base.extra, "lora_weight_name": ""},
        }
    )
    with pytest.raises(ModelConfigurationError, match="configured together"):
        ImageGenerationConfig.from_adapter_config(missing_weight)

    invalid_geometry = SimpleNamespace(
        **{
            **base.__dict__,
            "extra": {
                **base.extra,
                "preferred_generation_resolution": {"width": 1000, "height": 1024},
            },
        }
    )
    with pytest.raises(ModelConfigurationError, match="divisible by 16"):
        ImageGenerationConfig.from_adapter_config(invalid_geometry)


def test_image_profile_fingerprint_is_stable_and_registry_bound() -> None:
    config = ModelRegistry().role("t4_local", "image_generator")
    first = image_profile_fingerprint(config)
    second = image_profile_fingerprint(config)

    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == 71


def test_asset_request_uses_semantic_resource_contract_not_legacy_target_path() -> None:
    request = _asset()
    request.validate()

    assert request.visual_description == "blue lightsaber replacement"
    assert request.render_kind == "item.handheld"
    assert request.subject_id == "diamond_sword"
    assert request.requested_width == 16
    assert request.requested_height == 16
    assert not hasattr(request, "prompt")
    assert not hasattr(request, "target_path")


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
    assert sum(
        item["capability"] == "trade.transaction"
        for rows in ownership.values()
        for item in rows
    ) == 1
    assert sum(
        item["capability"] == "ui.shop_menu"
        for rows in ownership.values()
        for item in rows
    ) == 1
