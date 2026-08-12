from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from .fabric_official_template_provider import (
    FabricTemplateProviderError,
    bootstrap_fabric_project,
)
from .platform_catalog import adapter_for_lock_values, adapter_from_project


def install(orchestrator_module: Any) -> None:
    # Live module execution uses the same approved adapter all the way into the
    # coder prompt. Install this here because platform_api_contract is the late,
    # outermost platform hook and CustomModuleGenerator has already been imported.
    from . import custom_module_generator as custom_module_generator_module
    from .platform_custom_coder_contract import install as install_custom_coder

    install_custom_coder(custom_module_generator_module)

    cls = orchestrator_module.CompleteProductionOrchestrator
    original = cls._prepare_project
    if getattr(original, "_mmm_live_official_bootstrap", False):
        return

    @wraps(original)
    def prepare_project(
        self: Any,
        approved: Any,
        *,
        run_root: Path,
        existing_input: Any,
    ) -> Path:
        adapter = adapter_for_lock_values(approved.base_proposal.spec.platform)
        if existing_input is not None or adapter.source_api_family != "fabric_live_ai":
            return original(
                self,
                approved,
                run_root=run_root,
                existing_input=existing_input,
            )

        base = approved.base_proposal
        base.approve(base.calculate_hash())
        project_root = run_root / "base/workspaces" / base.spec.mod_id
        if project_root.exists():
            if _matches_live_target(self, project_root, base.spec, adapter):
                self._write_base_proposal(project_root, base)
                return project_root.resolve()
            self._preserve_partial_project(project_root)

        staging = project_root.with_name(f".{project_root.name}.staging")
        if staging.exists():
            self._preserve_partial_project(staging)
        try:
            receipt = bootstrap_fabric_project(
                project_root=staging,
                spec=base.spec,
                adapter=adapter,
                cache_root=run_root / ".cache/platform-bootstrap",
            )
        except FabricTemplateProviderError as exc:
            raise orchestrator_module.CompleteProductionError(
                "Official Fabric project bootstrap failed: " + str(exc)
            ) from exc

        self._write_base_proposal(staging, base)
        metadata = staging / ".minecraft_ai"
        metadata.mkdir(parents=True, exist_ok=True)
        import json

        (metadata / "fabric-template-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            actual = adapter_from_project(staging)
        except ValueError as exc:
            raise orchestrator_module.CompleteProductionError(
                "Official Fabric bootstrap could not be rebound to the approved target: "
                + str(exc)
            ) from exc
        if actual.minecraft_version != adapter.minecraft_version or actual.loader != adapter.loader:
            raise orchestrator_module.CompleteProductionError(
                "Official Fabric bootstrap target changed after approval."
            )

        if project_root.exists():
            self._preserve_partial_project(project_root)
        staging.replace(project_root)
        return project_root.resolve()

    prepare_project._mmm_live_official_bootstrap = True
    cls._prepare_project = prepare_project


def _matches_live_target(self: Any, root: Path, spec: Any, adapter: Any) -> bool:
    if not self._project_matches_spec(root, spec):
        return False
    try:
        actual = adapter_from_project(root)
    except Exception:
        return False
    return (
        actual.minecraft_version == adapter.minecraft_version
        and actual.loader == adapter.loader
        and actual.java_version == adapter.java_version
        and actual.fabric_loader == adapter.fabric_loader
        and actual.fabric_api == adapter.fabric_api
        and actual.fabric_loom == adapter.fabric_loom
        and actual.gradle == adapter.gradle
    )
