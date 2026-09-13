from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import resource_asset_production as assets
from minecraft_mod_ai.complete_spec import AssetRequest


def _request(*, width: int = 16, height: int = 16) -> AssetRequest:
    return AssetRequest(
        asset_id="widget_icon",
        kind="item",
        visual_description="A compact pixel-art widget icon",
        render_kind="item_texture",
        subject_id="widget",
        requested_width=width,
        requested_height=height,
    )


def test_semantic_asset_request_is_bound_to_current_resource_contract() -> None:
    request = _request()
    request.validate()
    assert request.asset_id == "widget_icon"
    assert request.kind == "item"
    assert request.render_kind == "item_texture"
    assert request.subject_id == "widget"
    assert request.requested_width == 16
    assert request.requested_height == 16

    with pytest.raises(Exception, match="requested dimensions|width|height|resource policy"):
        _request(width=16, height=0).validate()


def test_resource_manifest_rejects_conflicting_exact_target_paths() -> None:
    rows = [
        {
            "asset_id": "widget_icon",
            "container": "mod",
            "textures": [
                {
                    "target_path": "src/main/resources/assets/example/textures/item/widget.png",
                    "role": "base",
                }
            ],
            "documents": [],
        },
        {
            "asset_id": "other_icon",
            "container": "mod",
            "textures": [
                {
                    "target_path": "src/main/resources/assets/example/textures/item/widget.png",
                    "role": "base",
                }
            ],
            "documents": [],
        },
    ]
    with pytest.raises(assets.AssetProductionError, match="Conflicting resource manifest path"):
        assets._validate_manifest(rows)


def test_reference_closure_resolves_generated_texture_and_model_references(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_root = tmp_path / "run"
    namespace = "example"

    texture = project_root / "src/main/resources/assets/example/textures/item/widget.png"
    parent_model = project_root / "src/main/resources/assets/example/models/item/base.json"
    texture.parent.mkdir(parents=True)
    parent_model.parent.mkdir(parents=True)
    texture.write_bytes(b"generated-texture")
    parent_model.write_text("{}\n", encoding="utf-8")

    rows = [
        {
            "asset_id": "widget_icon",
            "container": "mod",
            "textures": [],
            "documents": [
                {
                    "target_path": "src/main/resources/assets/example/models/item/widget.json",
                    "payload": {
                        "parent": "example:item/base",
                        "textures": {"layer0": "example:item/widget"},
                    },
                }
            ],
        }
    ]

    report = assets._validate_reference_closure(
        project_root,
        run_root,
        rows,
        namespace=namespace,
    )

    assert report["status"] == "PASS"
    assert report["checked_reference_count"] == 2
    assert {item["kind"] for item in report["references"]} == {"model", "texture"}
