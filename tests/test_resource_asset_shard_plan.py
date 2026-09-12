from types import SimpleNamespace

from minecraft_mod_ai.resource_asset_plan import select_plan_rows


def test_select_plan_rows_preserves_requested_asset_order_and_subset():
    plan = {
        "assets": [
            {"asset_id": "texture_item_alpha", "textures": []},
            {"asset_id": "texture_item_beta", "textures": []},
            {"asset_id": "texture_item_gamma", "textures": []},
        ]
    }
    requests = (
        SimpleNamespace(asset_id="texture_item_gamma"),
        SimpleNamespace(asset_id="texture_item_alpha"),
    )

    selected = select_plan_rows(plan, requests)

    assert [row["asset_id"] for row in selected] == [
        "texture_item_gamma",
        "texture_item_alpha",
    ]


def test_select_plan_rows_fails_closed_when_requested_asset_is_missing():
    plan = {"assets": [{"asset_id": "texture_item_alpha", "textures": []}]}
    requests = (SimpleNamespace(asset_id="texture_item_missing"),)

    try:
        select_plan_rows(plan, requests)
    except RuntimeError as exc:
        assert "texture_item_missing" in str(exc)
    else:
        raise AssertionError("missing shard asset must fail closed")
