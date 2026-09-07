from __future__ import annotations

"""Pure deterministic contract for resource-asset generation inputs."""

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


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


__all__ = [
    "ResourceAssetPreflightError",
    "canonical_asset_target",
    "safe_asset_target",
    "validate_asset_generation_inputs",
]
