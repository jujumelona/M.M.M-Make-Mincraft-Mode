from types import SimpleNamespace

import pytest

from minecraft_mod_ai.complete_spec import AssetRequest, ProductionModule
from minecraft_mod_ai.deterministic_minecraft_content_contract import _compile_modules, _tool_schema
from minecraft_mod_ai.minecraft_generation_contract import (
    install_generation_guard,
    lower_generation_config,
    required_generation_fields,
    validate_generation_config,
)
from minecraft_mod_ai.minecraft_generation_design import _asset_main_color


def test_item_contract_uses_canonical_design_fields_not_generator_aliases():
    assert required_generation_fields("item") == ("display_name", "main_color")
    lowered = lower_generation_config(
        "item",
        "crystal",
        {"display_name": "Crystal", "main_color": "#12Ab34"},
    )
    assert lowered["display_name_en"] == "Crystal"
    assert lowered["display_name_ko"] == "Crystal"
    assert lowered["color"] == "#12Ab34"


def test_missing_semantic_generation_field_fails_closed():
    with pytest.raises(ValueError, match="MINECRAFT_GENERATION_FIELDS_REQUIRED"):
        validate_generation_config("block", "ore", {"display_name": "Ore", "main_color": "#112233"})


def test_data_only_artifacts_require_explicit_json_boundary():
    for kind in ("recipe", "advancement", "loot"):
        assert required_generation_fields(kind) == ("json",)
        with pytest.raises(ValueError, match="MINECRAFT_GENERATION_FIELDS_REQUIRED"):
            validate_generation_config(kind, "example", {})
        validate_generation_config(kind, "example", {"json": {"type": "minecraft:test"}})


def test_invalid_minecraft_specific_values_are_rejected():
    with pytest.raises(ValueError, match="permission_level"):
        validate_generation_config(
            "command",
            "admin",
            {"literal": "admin", "message": "ok", "permission_level": 5},
        )
    with pytest.raises(ValueError, match="input_item"):
        validate_generation_config(
            "machine",
            "press",
            {
                "display_name": "Press",
                "main_color": "#334455",
                "input_item": "not_namespaced",
                "output_item": "minecraft:diamond",
                "output_count": 1,
                "processing_ticks": 20,
            },
        )


def test_conflicting_legacy_color_cannot_override_design_authority():
    with pytest.raises(ValueError, match="FIELD_CONFLICT"):
        lower_generation_config(
            "item",
            "crystal",
            {
                "display_name": "Crystal",
                "main_color": "#112233",
                "color": "#445566",
            },
        )


def test_asset_projection_recovers_existing_main_color_without_model_reinference():
    graph = {
        "assets": [
            AssetRequest(
                asset_id="texture_item_crystal",
                kind="item",
                visual_description="shape: shard, main_color: #A1B2C3, surface: glass",
                render_kind="item.generated",
                subject_id="crystal",
            )
        ]
    }
    assert _asset_main_color(graph, "crystal") == "#A1B2C3"


def test_direct_generator_guard_lowers_before_delegate():
    seen = {}

    def generate_extended_content(*, modules, **kwargs):
        seen["modules"] = tuple(modules)
        return {"status": "ok"}

    fake = SimpleNamespace(
        _SUPPORTED={"item"},
        generate_extended_content=generate_extended_content,
    )
    install_generation_guard(fake)
    result = fake.generate_extended_content(
        modules=(
            ProductionModule(
                "crystal",
                "item",
                {"display_name": "Crystal", "main_color": "#123456"},
            ),
        )
    )
    assert result == {"status": "ok"}
    config = seen["modules"][0].config
    assert config["color"] == "#123456"
    assert config["display_name_en"] == "Crystal"


def test_small_model_tool_schema_requires_kind_specific_config_fields():
    fake = SimpleNamespace(_SUPPORTED={"item", "block"})
    schema = _tool_schema(fake)["function"]["parameters"]
    item_schema = schema["properties"]["modules"]["items"]
    conditions = item_schema["allOf"]
    item_condition = next(
        condition for condition in conditions if condition["if"]["properties"]["kind"]["const"] == "item"
    )
    assert item_condition["then"]["properties"]["config"]["required"] == [
        "display_name",
        "main_color",
    ]


def test_compile_modules_rejects_bare_module_and_accepts_canonical_item():
    fake = SimpleNamespace(
        _SUPPORTED={"item"},
        ProductionModule=ProductionModule,
    )
    with pytest.raises(ValueError, match="MINECRAFT_GENERATION_FIELDS_REQUIRED"):
        _compile_modules(fake, {"modules": [{"id": "crystal", "kind": "item"}]})

    compiled = _compile_modules(
        fake,
        {
            "modules": [
                {
                    "id": "crystal",
                    "kind": "item",
                    "config": {"display_name": "Crystal", "main_color": "#123456"},
                }
            ]
        },
    )
    assert compiled[0].config["display_name"] == "Crystal"
