from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from minecraft_mod_ai.complete_orchestrator_services import blockbench_review
from minecraft_mod_ai.complete_spec import AssetRequest
from minecraft_mod_ai.resource_asset_production import _model_size, _prepare
from minecraft_mod_ai.resource_contracts import resolve_asset


def _resolved_texture(width: int, height: int):
    asset = AssetRequest(
        asset_id="texture_gui_citadel",
        kind="gui",
        visual_description="A continuous frozen citadel panel",
        render_kind="gui.sprite",
        subject_id="citadel",
        requested_width=width,
        requested_height=height,
    )
    return resolve_asset(
        asset,
        namespace="test",
        minecraft_version="1.21.4",
    ).textures[0]


def test_large_explicit_resource_preserves_final_dimensions_after_backend_normalization(
    tmp_path: Path,
) -> None:
    texture = _resolved_texture(2305, 1301)
    assert (texture.width, texture.height) == (2305, 1301)
    assert texture.size_policy == "explicit"
    assert _model_size(texture.width, texture.height) == (1024, 1024)

    source = tmp_path / "source.png"
    normalized = tmp_path / "normalized.png"
    Image.new("RGBA", (1024, 1024), (12, 34, 56, 255)).save(source)

    score = _prepare(texture.to_dict(), source, normalized)

    assert score == 1000.0
    with Image.open(normalized) as image:
        assert image.size == (2305, 1301)


def test_small_explicit_resource_uses_backend_minimum_but_exact_final_dimensions(
    tmp_path: Path,
) -> None:
    texture = _resolved_texture(17, 31)
    assert (texture.width, texture.height) == (17, 31)
    assert _model_size(texture.width, texture.height) == (256, 256)

    source = tmp_path / "source.png"
    normalized = tmp_path / "normalized.png"
    Image.new("RGBA", (256, 256), (1, 2, 3, 255)).save(source)

    _prepare(texture.to_dict(), source, normalized)

    with Image.open(normalized) as image:
        assert image.size == (17, 31)


def test_backend_source_size_is_bounded_and_aligned():
    for width, height in ((1, 1), (17, 31), (257, 513), (2305, 1301)):
        source_width, source_height = _model_size(width, height)
        assert 256 <= source_width <= 1024
        assert 256 <= source_height <= 1024
        assert source_width % 16 == 0
        assert source_height % 16 == 0


def test_blockbench_review_scopes_client_to_the_run_root(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_root = tmp_path / "run"
    geometry = run_root / "project/model/test.geo.json"
    geometry.parent.mkdir(parents=True)
    geometry.write_text("{}", encoding="utf-8")
    seen: dict[str, Any] = {}

    class _FakeBlockbenchClient:
        def __init__(self, *, workspace_root: Path) -> None:
            seen["workspace_root"] = workspace_root

        def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
            if operation == "validate_uv":
                return {"status": "PASS"}
            if operation == "render_preview":
                preview = Path(arguments["output_path"])
                preview.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGBA", (16, 16), "blue").save(preview)
                return {"status": "PASS"}
            return {"status": "OK"}

        def close(self) -> None:
            seen["closed"] = True

    monkeypatch.setattr(
        "minecraft_mod_ai.complete_orchestrator_services.BlockbenchMCPClient",
        _FakeBlockbenchClient,
    )

    receipt = blockbench_review(
        {
            "entity_id": "test",
            "files": [str(geometry)],
        },
        run_root,
    )

    assert seen == {"workspace_root": run_root, "closed": True}
    assert Path(receipt["preview"]).is_file()
