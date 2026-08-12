from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any

from .fabric_official_template_provider import (
    FabricTemplateProviderError,
    bootstrap_fabric_project,
)
from .platform_catalog import adapter_for_lock_values, adapter_from_project


def install(orchestrator_module: Any) -> None:
    """Install only live-target project preparation and migration behavior."""
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
        selection = approved.game_design.get("_platform_selection", {})
        migration_requested = bool(
            existing_input is not None
            and isinstance(selection, dict)
            and selection.get("migration_requested")
        )

        if migration_requested:
            if adapter.source_api_family != "fabric_live_ai":
                raise orchestrator_module.CompleteProductionError(
                    "Version migration currently requires a live-discovered Fabric target."
                )
            return _prepare_live_migration(
                self,
                orchestrator_module,
                original,
                approved=approved,
                run_root=run_root,
                existing_input=existing_input,
                adapter=adapter,
                selection=selection,
            )

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


def _prepare_live_migration(
    self: Any,
    orchestrator_module: Any,
    wrapped_prepare: Any,
    *,
    approved: Any,
    run_root: Path,
    existing_input: Any,
    adapter: Any,
    selection: dict[str, Any],
) -> Path:
    """Import the old source, bind approved migration intent, then let AI port it."""
    try:
        report = orchestrator_module.inspect_existing_project_archive(existing_input)
    except Exception as exc:
        raise orchestrator_module.CompleteProductionError(
            "Could not inspect the Revise migration input: " + str(exc)
        ) from exc

    if report.loader and str(report.loader).strip().lower() != "fabric":
        raise orchestrator_module.CompleteProductionError(
            "Automatic version migration currently preserves the Fabric loader; "
            f"existing loader is {report.loader!r}."
        )
    migration_from = selection.get("migration_from")
    if isinstance(migration_from, dict):
        expected_from_version = str(migration_from.get("minecraft_version", "")).strip()
        expected_from_loader = str(migration_from.get("loader", "fabric")).strip().lower()
        if report.minecraft_version and expected_from_version and report.minecraft_version != expected_from_version:
            raise orchestrator_module.CompleteProductionError(
                "Approved migration source version does not match the bound Revise ZIP: "
                f"plan={expected_from_version}, input={report.minecraft_version}."
            )
        if report.loader and expected_from_loader and str(report.loader).lower() != expected_from_loader:
            raise orchestrator_module.CompleteProductionError(
                "Approved migration source loader does not match the bound Revise ZIP."
            )

    inner_prepare = getattr(wrapped_prepare, "__wrapped__", None)
    if not callable(inner_prepare):
        raise orchestrator_module.CompleteProductionError(
            "Migration preparation chain is not inspectable; refusing to bypass target guards."
        )
    root = inner_prepare(
        self,
        approved,
        run_root=run_root,
        existing_input=existing_input,
    )
    root = Path(root).resolve()
    self._write_base_proposal(root, approved.base_proposal)

    metadata = root / ".minecraft_ai"
    metadata.mkdir(parents=True, exist_ok=True)
    source = {
        "minecraft_version": str(report.minecraft_version or "unknown"),
        "loader": str(report.loader or "fabric"),
    }
    target = {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "java_version": adapter.java_version,
        "yarn_mappings": adapter.yarn_mappings,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "fabric_loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
    }
    receipt = {
        "schema_version": "mmm/platform-migration-intent-v1",
        "status": "PENDING_AI_PORT_AND_COMPILE_VALIDATION",
        "source": source,
        "target": target,
        "existing_input_sha256": approved.existing_input_sha256,
        "approval_hash": approved.calculate_hash(),
        "required_gates": ["JDT", "Gradle", "GameTest", "JAR", "runtime"],
    }
    (metadata / "platform-migration-intent.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_approved_target_lock(metadata, adapter, receipt)
    return root


def _write_approved_target_lock(metadata: Path, adapter: Any, receipt: dict[str, Any]) -> None:
    payload = {
        "schema_version": "mmm/generated-platform-lock-v2",
        "adapter_id": adapter.adapter_id,
        "edition": adapter.edition,
        "loader": adapter.loader,
        "minecraft_version": adapter.minecraft_version,
        "java_version": adapter.java_version,
        "yarn_mappings": adapter.yarn_mappings,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "fabric_loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
        "gradle_sha256": adapter.gradle_sha256,
        "source_api_family": adapter.source_api_family,
        "migration": receipt,
    }
    (metadata / "platform-lock.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
