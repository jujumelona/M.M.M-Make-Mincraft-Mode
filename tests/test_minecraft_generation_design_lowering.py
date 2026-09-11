from minecraft_mod_ai.complete_spec import AssetRequest, ProductionModule
from minecraft_mod_ai.minecraft_generation_design import complete_generation_fields


def test_completed_item_lowers_canonical_color_for_artifact_and_extended_paths():
    graph = {
        "modules": [
            ProductionModule(
                "crystal",
                "item",
                {"name": "Crystal"},
            )
        ],
        "assets": [
            AssetRequest(
                asset_id="texture_item_crystal",
                kind="item",
                prompt="Pixel Art, isolated, main_color: #A1B2C3, surface: glass",
                target_path="assets/example/textures/item/crystal.png",
            )
        ],
    }

    completed = complete_generation_fields(graph, None, prompt="Crystal mod")
    config = completed["modules"][0].config

    assert config["display_name"] == "Crystal"
    assert config["display_name_en"] == "Crystal"
    assert config["display_name_ko"] == "Crystal"
    assert config["main_color"] == "#A1B2C3"
    assert config["color"] == "#A1B2C3"


def test_data_only_module_is_not_asked_for_model_authored_json():
    module = ProductionModule(
        "crystal_recipe",
        "recipe",
        {"requirement_refs": ["req_crystal"], "reason": "craft crystal"},
    )
    completed = complete_generation_fields(
        {"modules": [module], "assets": []},
        None,
        prompt="Crystal mod",
    )
    assert completed["modules"][0] is module
    assert "json" not in completed["modules"][0].config
