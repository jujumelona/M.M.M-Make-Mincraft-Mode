from __future__ import annotations

"""Reconcile deterministic generation boundaries after runtime composition.

Game design is host-owned and deterministic. Runtime finalization installs the final
host-side generation wrappers here so stale imported callables cannot bypass approval
or generation preflight contracts.
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


def _install_geckolib_project_preflight(geckolib_module: Any) -> None:
    """Validate late-known project invariants at GeckoLib's last pre-write boundary."""

    from .geckolib_generation_contract import (
        GeckoLibGenerationContractError,
        validate_geckolib_project_preflight,
    )

    original_inspect = geckolib_module.inspect_fabric_project
    if getattr(original_inspect, "_mmm_geckolib_generation_preflight", False):
        return

    @wraps(original_inspect)
    def inspect_fabric_project(project_root: str | Path):
        info = original_inspect(project_root)
        try:
            validate_geckolib_project_preflight(info)
        except GeckoLibGenerationContractError as exc:
            raise geckolib_module.GeckoLibGenerationError(
                f"GeckoLib generation preflight failed: {exc}"
            ) from exc
        return info

    inspect_fabric_project._mmm_geckolib_generation_preflight = True
    inspect_fabric_project.__wrapped__ = original_inspect
    geckolib_module.inspect_fabric_project = inspect_fabric_project


def _install_production_generation_preflight(complete_spec_module: Any) -> None:
    """Reject proposal-known built-in generator failures before work dispatch."""

    from .production_generation_preflight import (
        ProductionGenerationPreflightError,
        validate_production_generation_modules,
    )
    from .spec import SpecValidationError

    original_validate = complete_spec_module.CompleteProposal.validate
    if getattr(original_validate, "_mmm_production_generation_preflight", False):
        return

    @wraps(original_validate)
    def validate(proposal: Any, *, policy: Any = None) -> None:
        original_validate(proposal, policy=policy)
        try:
            validate_production_generation_modules(
                proposal.modules,
                policy=policy,
            )
        except ProductionGenerationPreflightError as exc:
            raise SpecValidationError(
                f"Production generation preflight failed: {exc}"
            ) from exc

    validate._mmm_production_generation_preflight = True
    validate.__wrapped__ = original_validate
    complete_spec_module.CompleteProposal.validate = validate


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import complete_orchestrator, complete_spec
    from . import fabric_official_template_provider as fabric_provider
    from . import geckolib_generator, resource_asset_production

    _install_resource_asset_preflight(
        resource_asset_production,
        orchestrator_module=complete_orchestrator,
    )
    _install_geckolib_project_preflight(geckolib_generator)
    _install_production_generation_preflight(complete_spec)

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
