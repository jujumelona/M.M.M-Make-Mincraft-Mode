from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.resource_asset_preflight_contract import (
    ResourceAssetPreflightError,
    canonical_asset_target,
    install,
    safe_asset_target,
    validate_asset_generation_inputs,
)
from minecraft_mod_ai.spec import SpecValidationError


class _Proposal:
    def __init__(self, target_path: str, *, pack_format: object = 15) -> None:
        self.assets = (SimpleNamespace(target_path=target_path),)
        target = {} if pack_format is None else {"resource_pack_format": pack_format}
        self.game_design = {"_platform_selection": {"target": target}}
        self.validate_calls = 0

    def validate(self) -> None:
        self.validate_calls += 1


def test_canonical_target_rejects_generation_incompatible_paths() -> None:
    with pytest.raises(ResourceAssetPreflightError, match="non-PNG"):
        canonical_asset_target("assets/example/textures/item/test.json")
    with pytest.raises(ResourceAssetPreflightError, match="under assets"):
        canonical_asset_target("generated/test.png")
    with pytest.raises(ResourceAssetPreflightError, match="non-PNG"):
        canonical_asset_target("../assets/example/test.png")


def test_preflight_requires_pack_format_before_standalone_generation() -> None:
    proposal = _Proposal("assets/example/textures/item/test.png", pack_format=None)

    with pytest.raises(ResourceAssetPreflightError, match="resource_pack_format"):
        validate_asset_generation_inputs(proposal)

    assert proposal.validate_calls == 1


def test_mod_resource_target_does_not_require_standalone_pack_format() -> None:
    proposal = _Proposal(
        "src/main/resources/assets/example/textures/item/test.png",
        pack_format=None,
    )

    assert validate_asset_generation_inputs(proposal) == {}


def test_safe_target_uses_same_canonical_path_contract(tmp_path: Path) -> None:
    target = safe_asset_target(
        tmp_path,
        "src/main/resources/assets/example/textures/item/test.png",
    )
    assert target == (
        tmp_path / "src/main/resources/assets/example/textures/item/test.png"
    ).resolve()


def test_installed_prompt_preflight_blocks_original_planner_call() -> None:
    calls = {"planner": 0}

    def original_attach(_router, proposal):
        calls["planner"] += 1
        return proposal

    def original_generate(_router, _proposal, _project_root, _run_root):
        raise AssertionError("image generation must not be reached")

    fake_module = SimpleNamespace(
        attach_generation_plan=original_attach,
        generate_assets=original_generate,
        AssetProductionError=RuntimeError,
        _safe_target=lambda *_args: None,
    )
    install(fake_module)
    proposal = _Proposal("assets/example/textures/item/test.json")

    with pytest.raises(SpecValidationError, match="Resource asset preflight failed"):
        fake_module.attach_generation_plan(object(), proposal)

    assert calls["planner"] == 0


def test_installed_binary_preflight_blocks_original_image_generation(tmp_path: Path) -> None:
    calls = {"image": 0}

    def original_attach(_router, proposal):
        return proposal

    def original_generate(_router, _proposal, _project_root, _run_root):
        calls["image"] += 1
        return {"status": "unexpected"}

    fake_module = SimpleNamespace(
        attach_generation_plan=original_attach,
        generate_assets=original_generate,
        AssetProductionError=RuntimeError,
        _safe_target=lambda *_args: None,
    )
    install(fake_module)
    proposal = _Proposal("assets/example/textures/item/test.png", pack_format=None)

    with pytest.raises(RuntimeError, match="Resource asset preflight failed"):
        fake_module.generate_assets(object(), proposal, tmp_path, tmp_path)

    assert calls["image"] == 0
