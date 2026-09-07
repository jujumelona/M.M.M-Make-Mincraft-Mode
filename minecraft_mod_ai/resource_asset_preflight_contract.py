from __future__ import annotations

"""Fail closed on deterministic resource-asset inputs before model generation."""

from collections.abc import Mapping
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

from .spec import SpecValidationError


class ResourceAssetPreflightError(ValueError):
    """The approved proposal cannot reach the resource producer deterministically."""


def canonical_asset_target(raw_path: str) -> PurePosixPath:
    """Return the canonical Minecraft PNG target or reject it before generation."""

    normalized = str(raw_path).replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or pure.suffix.casefold() != ".png":
        raise ResourceAssetPreflightError(
            f"Unsafe/non-PNG Minecraft asset target: {raw_path!r}"
        )
    if "assets" not in pure.parts:
        raise ResourceAssetPreflightError(
            f"Minecraft texture target must live under assets/: {raw_path!r}"
        )
    return pure


def _target_receipt(game_design: Any) -> dict[str, Any]:
    selection = game_design.get("_platform_selection") if isinstance(game_design, Mapping) else None
    target = selection.get("target") if isinstance(selection, Mapping) else None
    return dict(target) if isinstance(target, Mapping) else {}


def validate_asset_generation_inputs(proposal: Any) -> dict[str, Any]:
    """Validate every input condition knowable before prompt/image model calls."""

    validator = getattr(proposal, "validate", None)
    if not callable(validator):
        raise ResourceAssetPreflightError("Resource asset generation requires a validated proposal.")
    validator()

    assets = tuple(getattr(proposal, "assets", ()) or ())
    target = _target_receipt(getattr(proposal, "game_design", None))
    standalone = False
    for asset in assets:
        pure = canonical_asset_target(str(getattr(asset, "target_path", "")))
        standalone = standalone or bool(pure.parts and pure.parts[0] == "assets")

    if standalone:
        pack_format = target.get("resource_pack_format")
        if type(pack_format) is not int or pack_format < 1:
            raise ResourceAssetPreflightError(
                "Selected platform must supply a positive resource_pack_format before asset generation."
            )
    return target


def safe_asset_target(project_root: Path, raw_path: str) -> Path:
    """Resolve one canonical asset target under the project root."""

    pure = canonical_asset_target(raw_path)
    root = Path(project_root).expanduser().resolve()
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ResourceAssetPreflightError(
            f"Minecraft asset target escaped the project root: {raw_path!r}"
        ) from exc
    return target


def install(resource_module: Any) -> None:
    """Install one shared preflight in both prompt planning and binary generation."""

    original_attach = resource_module.attach_generation_plan
    if not getattr(original_attach, "_mmm_resource_asset_preflight", False):

        @wraps(original_attach)
        def attach_generation_plan(router: Any, proposal: Any):
            if getattr(proposal, "assets", None):
                try:
                    validate_asset_generation_inputs(proposal)
                except ResourceAssetPreflightError as exc:
                    raise SpecValidationError(f"Resource asset preflight failed: {exc}") from exc
            return original_attach(router, proposal)

        attach_generation_plan._mmm_resource_asset_preflight = True
        attach_generation_plan.__wrapped__ = original_attach
        resource_module.attach_generation_plan = attach_generation_plan

    original_generate = resource_module.generate_assets
    if not getattr(original_generate, "_mmm_resource_asset_preflight", False):

        @wraps(original_generate)
        def generate_assets(router: Any, proposal: Any, project_root: Path, run_root: Path):
            if hasattr(proposal, "game_design") and getattr(proposal, "assets", None):
                try:
                    validate_asset_generation_inputs(proposal)
                except (ResourceAssetPreflightError, SpecValidationError) as exc:
                    raise resource_module.AssetProductionError(
                        f"Resource asset preflight failed: {exc}"
                    ) from exc
            return original_generate(router, proposal, project_root, run_root)

        generate_assets._mmm_resource_asset_preflight = True
        generate_assets.__wrapped__ = original_generate
        resource_module.generate_assets = generate_assets

    def safe_target(project_root: Path, raw_path: str) -> Path:
        try:
            return safe_asset_target(project_root, raw_path)
        except ResourceAssetPreflightError as exc:
            raise resource_module.AssetProductionError(str(exc)) from exc

    safe_target._mmm_resource_asset_preflight = True
    resource_module._safe_target = safe_target


__all__ = [
    "ResourceAssetPreflightError",
    "canonical_asset_target",
    "install",
    "safe_asset_target",
    "validate_asset_generation_inputs",
]
