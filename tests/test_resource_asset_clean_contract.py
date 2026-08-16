from __future__ import annotations

import json

from minecraft_mod_ai.complete_spec import AssetRequest
from minecraft_mod_ai import resource_asset_production as assets


def _request(*, path: str = "assets/example/textures/item/widget.png", width: int = 16, height: int = 16) -> AssetRequest:
    return AssetRequest(
        asset_id="widget_icon",
        kind="item",
        prompt="replace the widget",
        target_path=path,
        width=width,
        height=height,
    )


def _plan(request: AssetRequest) -> dict:
    return {
        "schema_version": assets.PLAN_SCHEMA,
        "model_id": assets.FLUX_MODEL_ID,
        "quantization": assets.QUANTIZATION,
        "lora_model_id": assets.PIXEL_LORA_ID,
        "lora_weight_name": assets.PIXEL_LORA_WEIGHT,
        "lora_trigger": assets.PIXEL_LORA_TRIGGER,
        "prompt_candidates_per_asset": 4,
        "assets": [
            {
                "asset_id": request.asset_id,
                "kind": request.kind,
                "target_path": request.target_path,
                "width": request.width,
                "height": request.height,
                "prompts": [
                    f"{assets.PIXEL_LORA_TRIGGER}. candidate {index} widget replacement"
                    for index in range(4)
                ],
            }
        ],
    }


def test_persisted_asset_plan_is_bound_to_exact_resource_contract() -> None:
    request = _request()
    plan = _plan(request)
    assert assets._valid_plan(plan, (request,)) is True

    changed_path = _request(path="assets/example/textures/item/other.png")
    changed_size = _request(width=32, height=32)
    assert assets._valid_plan(plan, (changed_path,)) is False
    assert assets._valid_plan(plan, (changed_size,)) is False

    drifted_kind = _plan(request)
    drifted_kind["assets"][0]["kind"] = "icon"
    assert assets._valid_plan(drifted_kind, (request,)) is False

    duplicate_prompts = _plan(request)
    duplicate_prompts["assets"][0]["prompts"][3] = duplicate_prompts["assets"][0]["prompts"][0]
    assert assets._valid_plan(duplicate_prompts, (request,)) is False

    malformed_prompts = _plan(request)
    malformed_prompts["assets"][0]["prompts"][0] = {"unexpected": "object"}
    assert assets._valid_plan(malformed_prompts, (request,)) is False


def test_reference_closure_parses_each_json_once(monkeypatch, tmp_path) -> None:
    namespace = tmp_path / "assets" / "example"
    model = namespace / "models" / "item" / "widget.json"
    texture = namespace / "textures" / "item" / "widget.png"
    model.parent.mkdir(parents=True)
    texture.parent.mkdir(parents=True)
    model.write_text(
        json.dumps(
            {
                "parent": "minecraft:item/generated",
                "textures": {"layer0": "example:item/widget"},
            }
        ),
        encoding="utf-8",
    )
    texture.write_bytes(b"not-decoded-here")

    calls = 0
    original_loads = assets.json.loads

    def counted_loads(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(assets.json, "loads", counted_loads)
    report = assets._validate_reference_closure(tmp_path)

    assert report["ok"] is True
    assert report["checked_texture_references"] == 1
    assert report["checked_model_references"] == 1
    assert calls == 1
