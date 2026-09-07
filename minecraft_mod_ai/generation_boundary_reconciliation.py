from __future__ import annotations

"""Reconcile deterministic generation boundaries after runtime composition.

Game design is host-owned and deterministic. Runtime finalization installs the final
host-side generation wrappers here so stale imported callables cannot bypass approval
or resource-asset preflight contracts.
"""

import json
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALLED = False


def _write_approval_bound_bootstrap_lock(
    root: Path,
    adapter: Any,
    receipt: dict[str, Any],
) -> None:
    """Use the canonical immutable writer, then attach bootstrap evidence."""
    from . import platform_generation_contract

    platform_generation_contract._write_platform_lock(root, adapter)
    target = Path(root) / ".minecraft_ai" / "platform-lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Canonical platform lock writer did not produce an object.")
    payload["bootstrap"] = dict(receipt)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _install_resource_asset_preflight(
    resource_module: Any,
    *,
    orchestrator_module: Any | None = None,
) -> None:
    """Own the final wrappers while delegating validation to the pure shared contract."""

    from .resource_asset_preflight_contract import (
        ResourceAssetPreflightError,
        safe_asset_target,
        validate_asset_generation_inputs,
    )
    from .spec import SpecValidationError

    original_attach = resource_module.attach_generation_plan
    if not getattr(original_attach, "_mmm_resource_asset_preflight", False):

        @wraps(original_attach)
        def attach_generation_plan(router: Any, proposal: Any):
            if getattr(proposal, "assets", None):
                try:
                    validate_asset_generation_inputs(proposal)
                except (ResourceAssetPreflightError, SpecValidationError) as exc:
                    raise SpecValidationError(
                        f"Resource asset preflight failed: {exc}"
                    ) from exc
            return original_attach(router, proposal)

        attach_generation_plan._mmm_resource_asset_preflight = True
        attach_generation_plan.__wrapped__ = original_attach
        resource_module.attach_generation_plan = attach_generation_plan

    def safe_target(project_root: Path, raw_path: str) -> Path:
        try:
            return safe_asset_target(project_root, raw_path)
        except ResourceAssetPreflightError as exc:
            raise resource_module.AssetProductionError(str(exc)) from exc

    safe_target._mmm_resource_asset_preflight = True
    resource_module._safe_target = safe_target

    def wrap_generate(owner: Any) -> None:
        original_generate = owner.generate_assets
        if getattr(original_generate, "_mmm_resource_asset_preflight", False):
            return

        @wraps(original_generate)
        def generate_assets(
            router: Any,
            proposal: Any,
            project_root: Path,
            run_root: Path,
        ):
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
        owner.generate_assets = generate_assets

    wrap_generate(resource_module)
    if orchestrator_module is not None and orchestrator_module is not resource_module:
        wrap_generate(orchestrator_module)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import complete_orchestrator
    from . import fabric_official_template_provider as fabric_provider
    from . import resource_asset_production

    _install_resource_asset_preflight(
        resource_asset_production,
        orchestrator_module=complete_orchestrator,
    )

    original_platform_lock_writer = fabric_provider._write_platform_lock
    if not getattr(original_platform_lock_writer, "_mmm_approval_bound_bootstrap_lock", False):

        @wraps(original_platform_lock_writer)
        def write_platform_lock(root: Path, adapter: Any, receipt: dict[str, Any]) -> None:
            _write_approval_bound_bootstrap_lock(root, adapter, receipt)

        write_platform_lock._mmm_approval_bound_bootstrap_lock = True
        write_platform_lock.__wrapped__ = original_platform_lock_writer
        fabric_provider._write_platform_lock = write_platform_lock

    _INSTALLED = True


__all__ = ["install"]
