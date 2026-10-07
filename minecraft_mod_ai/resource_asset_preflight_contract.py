from __future__ import annotations

"""Pure deterministic contract for resource-asset generation inputs."""

from collections.abc import Mapping
import os
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
        if hasattr(asset, "render_kind"):
            asset.validate()
            standalone = standalone or asset.container == "resource_pack"
        else:
            # Legacy boundary callers still receive strict PNG path validation.
            pure = canonical_asset_target(str(getattr(asset, "target_path", "")))
            standalone = standalone or bool(pure.parts and pure.parts[0] == "assets")

    if standalone:
        platform = getattr(getattr(getattr(proposal, "base_proposal", None), "spec", None), "platform", None)
        pack_format = getattr(platform, "resource_pack_format", target.get("resource_pack_format"))
        if type(pack_format) is not int or pack_format < 1:
            raise ResourceAssetPreflightError(
                "Selected platform must supply a positive resource_pack_format before asset generation."
            )
    return target


def _env_true(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _probe_hf_file_access(repo_id: str, filename: str) -> dict[str, str]:
    """Check one exact Hugging Face file without downloading model weights."""

    try:
        from huggingface_hub import (
            get_hf_file_metadata,
            hf_hub_url,
            try_to_load_from_cache,
        )
    except ImportError as exc:
        raise ResourceAssetPreflightError(
            "Image backend requires huggingface_hub before production starts."
        ) from exc

    offline = _env_true("HF_HUB_OFFLINE") or _env_true("TRANSFORMERS_OFFLINE")
    if offline:
        cached = try_to_load_from_cache(repo_id, filename)
        if not isinstance(cached, str) or not Path(cached).is_file():
            raise ResourceAssetPreflightError(
                "IMAGE_BACKEND_ACCESS_PREFLIGHT_FAILED: offline mode is enabled "
                f"but {repo_id}/{filename} is not cached."
            )
        return {"repo_id": repo_id, "filename": filename, "access": "cached"}

    try:
        get_hf_file_metadata(
            hf_hub_url(repo_id=repo_id, filename=filename),
            timeout=15,
        )
    except Exception as exc:
        raise ResourceAssetPreflightError(
            "IMAGE_BACKEND_ACCESS_PREFLIGHT_FAILED: cannot access required "
            f"Hugging Face file {repo_id}/{filename}: {type(exc).__name__}: {exc}. "
            "Authenticate/accept repository access or configure an accessible backend "
            "before production starts."
        ) from exc

    return {"repo_id": repo_id, "filename": filename, "access": "remote"}


def validate_image_backend_access(router: Any) -> dict[str, Any]:
    """Validate the configured image backend before production/model work."""

    registry = getattr(router, "registry", None)
    profile_name = str(getattr(router, "profile", "") or "").strip()
    if registry is None or not profile_name:
        raise ResourceAssetPreflightError(
            "Image backend preflight requires a configured model router."
        )

    try:
        config = registry.role(profile_name, "image_generator")
    except Exception as exc:
        raise ResourceAssetPreflightError(
            f"Image generator profile resolution failed: {type(exc).__name__}: {exc}"
        ) from exc

    adapter = str(getattr(config, "adapter", "") or "").strip()
    if adapter == "image_diffusion":
        from .model_adapters.base import require_package

        require_package("huggingface_hub", minimum="0.34.0")
        require_package("diffusers", minimum="0.39.0")
        require_package("transformers", minimum="4.56.0")
        require_package("accelerate", minimum="1.0.0")
        if getattr(config, "quantization", None):
            require_package("bitsandbytes", minimum="0.45.0")

        model_id = str(getattr(config, "model_id", "") or "").strip()
        if not model_id:
            raise ResourceAssetPreflightError("Image diffusion model_id is empty.")

        checks: list[dict[str, str]] = []
        local_model = Path(model_id).expanduser()
        if local_model.is_dir():
            model_index = local_model / "model_index.json"
            if not model_index.is_file():
                raise ResourceAssetPreflightError(
                    f"Local image model has no model_index.json: {local_model}"
                )
            checks.append({
                "repo_id": str(local_model),
                "filename": "model_index.json",
                "access": "local",
            })
        else:
            checks.append(_probe_hf_file_access(model_id, "model_index.json"))

        extra = getattr(config, "extra", {})
        extra = extra if isinstance(extra, Mapping) else {}
        lora_repo = str(extra.get("lora_model_id") or "").strip()
        lora_file = str(extra.get("lora_weight_name") or "").strip()

        if bool(lora_repo) != bool(lora_file):
            raise ResourceAssetPreflightError(
                "Image LoRA model_id and weight_name must be configured together."
            )

        if lora_repo:
            require_package("peft", minimum="0.17.0")
            local_lora = Path(lora_repo).expanduser()
            if local_lora.is_dir():
                weight = local_lora / lora_file
                if not weight.is_file():
                    raise ResourceAssetPreflightError(
                        f"Local image LoRA is missing: {weight}"
                    )
                checks.append({
                    "repo_id": str(local_lora),
                    "filename": lora_file,
                    "access": "local",
                })
            else:
                checks.append(_probe_hf_file_access(lora_repo, lora_file))

        return {
            "schema_version": "mmm/image-backend-access-preflight-v1",
            "status": "PASS",
            "adapter": adapter,
            "model_id": model_id,
            "checks": checks,
        }

    if adapter == "openai_compatible":
        if not (
            str(getattr(config, "model_id", "") or "").strip()
            and str(getattr(config, "base_url", "") or "").strip()
            and str(getattr(config, "api_key", "") or "").strip()
        ):
            raise ResourceAssetPreflightError(
                "Remote image backend is missing model/base_url/api_key configuration."
            )
        return {
            "schema_version": "mmm/image-backend-access-preflight-v1",
            "status": "PASS",
            "adapter": adapter,
            "model_id": str(config.model_id),
            "checks": [],
        }

    if adapter == "mock":
        return {
            "schema_version": "mmm/image-backend-access-preflight-v1",
            "status": "PASS",
            "adapter": adapter,
            "model_id": str(getattr(config, "model_id", "") or ""),
            "checks": [],
        }

    raise ResourceAssetPreflightError(
        f"Image generator uses unsupported adapter {adapter!r}."
    )


def validate_asset_backend_access(router: Any, proposal: Any) -> dict[str, Any]:
    """Validate image backend availability only when the proposal needs assets."""

    if not tuple(getattr(proposal, "assets", ()) or ()):
        return {
            "schema_version": "mmm/image-backend-access-preflight-v1",
            "status": "NOT_REQUIRED",
        }
    return validate_image_backend_access(router)


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
    "validate_asset_backend_access",
    "validate_asset_generation_inputs",
    "validate_image_backend_access",
]
