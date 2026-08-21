from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from minecraft_mod_ai.complete_orchestrator_services import (
    blockbench_review,
    generate_assets,
)
from minecraft_mod_ai.complete_spec import AssetRequest
from minecraft_mod_ai.source_patch import sha256_file


class _DeterministicImageRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @contextmanager
    def image_generation_session(self, role: str):
        assert role == "image_generator"
        yield

    def generate_image(
        self,
        role: str,
        *,
        prompt: str,
        output_path: str | Path,
        width: int,
        height: int,
        seed: int,
    ) -> Path:
        assert role == "image_generator"
        assert 256 <= width <= 1024
        assert 256 <= height <= 1024
        assert width % 16 == 0
        assert height % 16 == 0
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        color = (
            seed & 0xFF,
            (seed >> 8) & 0xFF,
            (seed >> 16) & 0xFF,
            255,
        )
        Image.new("RGBA", (width, height), color).save(path)
        self.calls.append(
            {
                "prompt": prompt,
                "path": str(path),
                "width": width,
                "height": height,
                "seed": seed,
            }
        )
        return path


def _large_asset_proposal() -> SimpleNamespace:
    return SimpleNamespace(
        assets=(
            AssetRequest(
                asset_id="large_environment",
                kind="environment",
                prompt="A continuous frozen citadel panorama",
                target_path=(
                    "src/main/resources/assets/test/textures/"
                    "environment/citadel.png"
                ),
                width=2305,
                height=1301,
            ),
        )
    )


def test_large_asset_uses_deterministic_overlapping_source_tiles(
    tmp_path: Path,
) -> None:
    first_router = _DeterministicImageRouter()
    first_project = tmp_path / "first-project"
    first_project.mkdir()
    first = generate_assets(
        first_router,
        _large_asset_proposal(),
        first_project,
        tmp_path / "first-run",
    )
    first_asset = first["assets"][0]
    target = Path(first_asset["target"])

    assert first["schema_version"] == "mmm/complete-assets-v3"
    assert first_asset["source_mode"] == "multiscale_overlapping_tiles"
    assert first_asset["width"] == 2305
    assert first_asset["height"] == 1301
    assert len(first_asset["source_tiles"]) > 1
    assert any(
        tile["left_overlap"] or tile["top_overlap"]
        for tile in first_asset["source_tiles"]
    )
    assert all(
        tile["source_width"] <= 1024
        and tile["source_height"] <= 1024
        for tile in first_asset["source_tiles"]
    )
    with Image.open(target) as image:
        assert image.size == (2305, 1301)
    assert first_asset["sha256"] == sha256_file(target)

    second_router = _DeterministicImageRouter()
    second_project = tmp_path / "second-project"
    second_project.mkdir()
    second = generate_assets(
        second_router,
        _large_asset_proposal(),
        second_project,
        tmp_path / "second-run",
    )
    second_asset = second["assets"][0]

    assert second_asset["sha256"] == first_asset["sha256"]
    assert [
        (call["width"], call["height"], call["seed"], call["prompt"])
        for call in second_router.calls
    ] == [
        (call["width"], call["height"], call["seed"], call["prompt"])
        for call in first_router.calls
    ]
    assert len(first_router.calls) == 1 + len(first_asset["source_tiles"])


def test_small_asset_keeps_exact_dimensions_without_high_resolution_tiling(
    tmp_path: Path,
) -> None:
    router = _DeterministicImageRouter()
    project = tmp_path / "project"
    project.mkdir()
    proposal = SimpleNamespace(
        assets=(
            AssetRequest(
                asset_id="small_icon",
                kind="icon",
                prompt="A blue crystal icon",
                target_path="src/main/resources/assets/test/icon.png",
                width=17,
                height=31,
            ),
        )
    )

    receipt = generate_assets(
        router,
        proposal,
        project,
        tmp_path / "run",
    )
    asset = receipt["assets"][0]

    assert asset["source_mode"] == "single_source"
    assert len(router.calls) == 1
    assert router.calls[0]["width"] == 256
    assert router.calls[0]["height"] == 256
    with Image.open(asset["target"]) as image:
        assert image.size == (17, 31)


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
