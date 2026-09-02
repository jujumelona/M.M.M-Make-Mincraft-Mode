from __future__ import annotations

"""Freeze the complete executable platform receipt across approval and execution.

Resolution may use live provider evidence. Once an adapter is selected, every coordinate
needed by generation/build validation is copied into the canonical approval-bound
``PlatformLock``. Downstream reconstruction uses only that receipt and therefore cannot
silently drift when Fabric/Minecraft metadata changes later.
"""

import json
import re
from functools import wraps
from pathlib import Path
from typing import Any

from .spec import PlatformLock, SpecValidationError, platform_receipt_sha256

_INSTALLED = False


def _receipt_value(value: Any, name: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _gradle_distribution_url(version: str) -> str:
    return f"https://services.gradle.org/distributions/gradle-{version}-bin.zip"


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = str(value or "").split(".")
    if not parts or not all(part.isdigit() for part in parts):
        return ()
    return tuple(int(part) for part in parts)


def _validate_base_coordinate_consistency(lock: PlatformLock) -> None:
    """Reject mixed target tuples even for older narrow persisted locks."""

    minecraft = str(lock.minecraft_version)
    java = str(lock.java_version)
    mc_tuple = _version_tuple(minecraft)
    if mc_tuple and mc_tuple[0] == 1 and mc_tuple >= (1, 20, 5):
        if java.isdigit() and int(java) < 21:
            raise SpecValidationError(
                f"Minecraft {minecraft} requires a Java 21+ execution target; got Java {java}."
            )

    yarn = str(lock.yarn_mappings)
    if yarn.casefold() not in {"mojang", "official", "official_mojang"}:
        mapping_match = re.match(r"^(\d+(?:\.\d+){1,2})\+", yarn)
        if mapping_match and mapping_match.group(1) != minecraft:
            raise SpecValidationError(
                "Platform lock mappings coordinate disagrees with minecraft_version."
            )

    api_match = re.search(r"\+(\d+(?:\.\d+){1,2})$", str(lock.fabric_api))
    if api_match and api_match.group(1) != minecraft:
        raise SpecValidationError(
            "Platform lock Fabric API coordinate disagrees with minecraft_version."
        )


def _full_receipt_present(value: Any) -> bool:
    if isinstance(value, PlatformLock):
        return value.has_full_execution_receipt()
    required_nonempty = (
        "adapter_id",
        "mappings_kind",
        "mappings_version",
        "gradle_sha256",
        "gradle_distribution_url",
        "data_pack_version",
        "resource_pack_version",
        "resource_pack_format",
        "release_metadata_url",
        "source_api_family",
        "receipt_sha256",
    )
    if not all(
        _receipt_value(value, name, None) not in (None, "", 0)
        for name in required_nonempty
    ):
        return False
    return _receipt_value(value, "deterministic_module_kinds", None) is not None


def _validate_receipt_digest(value: Any) -> None:
    supplied = str(_receipt_value(value, "receipt_sha256", "")).casefold()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", supplied):
        raise SpecValidationError(
            "Platform lock receipt_sha256 must be a full SHA-256 receipt."
        )
    expected = platform_receipt_sha256(value)
    if supplied != expected:
        raise SpecValidationError(
            "Platform lock receipt SHA-256 does not match its immutable provider coordinates."
        )


def _adapter_from_receipt(value: Any):
    from .platform_catalog import PlatformAdapter

    if not _full_receipt_present(value):
        raise SpecValidationError(
            "Execution platform receipt is incomplete; live provider re-resolution is forbidden."
        )
    if hasattr(value, "validate"):
        value.validate()
    else:
        _validate_receipt_digest(value)

    gradle = str(_receipt_value(value, "gradle"))
    expected_url = _gradle_distribution_url(gradle)
    if str(_receipt_value(value, "gradle_distribution_url")) != expected_url:
        raise SpecValidationError(
            "Execution platform receipt Gradle distribution URL disagrees with the pinned version."
        )
    sha = str(_receipt_value(value, "gradle_sha256")).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise SpecValidationError("Execution platform receipt Gradle SHA-256 is invalid.")

    adapter = PlatformAdapter(
        adapter_id=str(_receipt_value(value, "adapter_id")),
        edition=str(_receipt_value(value, "edition")),
        loader=str(_receipt_value(value, "loader")),
        minecraft_version=str(_receipt_value(value, "minecraft_version")),
        java_version=str(_receipt_value(value, "java_version")),
        yarn_mappings=str(_receipt_value(value, "yarn_mappings")),
        mappings_kind=str(_receipt_value(value, "mappings_kind")),
        mappings_version=str(_receipt_value(value, "mappings_version")),
        fabric_loader=str(_receipt_value(value, "fabric_loader")),
        fabric_api=str(_receipt_value(value, "fabric_api")),
        fabric_loom=str(_receipt_value(value, "fabric_loom")),
        gradle=gradle,
        gradle_sha256=sha,
        data_pack_version=str(_receipt_value(value, "data_pack_version")),
        resource_pack_version=str(_receipt_value(value, "resource_pack_version")),
        resource_pack_format=int(_receipt_value(value, "resource_pack_format")),
        release_metadata_url=str(_receipt_value(value, "release_metadata_url")),
        source_api_family=str(_receipt_value(value, "source_api_family")),
        deterministic_module_kinds=frozenset(
            str(item)
            for item in (_receipt_value(value, "deterministic_module_kinds", ()) or ())
            if str(item).strip()
        ),
    )
    adapter.validate()
    return adapter


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import (
        generator,
        platform_catalog,
        platform_generation_contract,
        platform_live_execution_contract,
        platform_resolver,
        platform_runtime_contract,
        platform_validation_contract,
        runner,
    )

    current_platform_validate = PlatformLock.validate
    if not getattr(current_platform_validate, "_mmm_base_coordinate_consistency", False):

        @wraps(current_platform_validate)
        def validate_platform_lock(self: PlatformLock) -> None:
            _validate_base_coordinate_consistency(self)
            current_platform_validate(self)

        validate_platform_lock._mmm_base_coordinate_consistency = True
        validate_platform_lock.__wrapped__ = current_platform_validate
        PlatformLock.validate = validate_platform_lock

    def lock_from_adapter(adapter):
        adapter.validate()
        values = {
            "edition": adapter.edition,
            "loader": adapter.loader,
            "minecraft_version": adapter.minecraft_version,
            "java_version": adapter.java_version,
            "yarn_mappings": adapter.yarn_mappings,
            "fabric_loader": adapter.fabric_loader,
            "fabric_api": adapter.fabric_api,
            "fabric_loom": adapter.fabric_loom,
            "gradle": adapter.gradle,
            "adapter_id": adapter.adapter_id,
            "mappings_kind": adapter.mappings_kind,
            "mappings_version": adapter.mappings_version,
            "gradle_sha256": adapter.gradle_sha256,
            "gradle_distribution_url": _gradle_distribution_url(adapter.gradle),
            "data_pack_version": adapter.data_pack_version,
            "resource_pack_version": adapter.resource_pack_version,
            "resource_pack_format": adapter.resource_pack_format,
            "release_metadata_url": adapter.release_metadata_url,
            "source_api_family": adapter.source_api_family,
            "deterministic_module_kinds": tuple(sorted(adapter.deterministic_module_kinds)),
        }
        values["receipt_sha256"] = platform_receipt_sha256(values)
        lock = PlatformLock(**values)
        lock.validate()
        return lock

    original_adapter_from_project = platform_catalog.adapter_from_project

    def adapter_for_lock_values(value):
        # This is an execution boundary. Missing receipt fields are never rediscovered
        # from mutable provider state after approval.
        if hasattr(value, "validate"):
            value.validate()
        return _adapter_from_receipt(value)

    def adapter_from_project(project_root):
        lock_path = Path(project_root) / ".minecraft_ai" / "platform-lock.json"
        if lock_path.is_file():
            raw = json.loads(lock_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not _full_receipt_present(raw):
                raise SpecValidationError(
                    "Generated platform lock is incomplete; re-plan instead of live re-resolution."
                )
            return _adapter_from_receipt(raw)
        # A foreign/existing project without an MMM lock may still be inspected before
        # approval. Once MMM writes a lock, that lock is authoritative and fail-closed.
        return original_adapter_from_project(project_root)

    def write_project_platform_lock(
        project_root: Path,
        adapter,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        target = Path(project_root) / ".minecraft_ai" / "platform-lock.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "mmm/generated-platform-lock-v4",
            "adapter_id": adapter.adapter_id,
            "edition": adapter.edition,
            "loader": adapter.loader,
            "minecraft_version": adapter.minecraft_version,
            "java_version": adapter.java_version,
            "yarn_mappings": adapter.yarn_mappings,
            "mappings_kind": adapter.mappings_kind,
            "mappings_version": adapter.mappings_version,
            "fabric_loader": adapter.fabric_loader,
            "fabric_api": adapter.fabric_api,
            "fabric_loom": adapter.fabric_loom,
            "gradle": adapter.gradle,
            "gradle_sha256": adapter.gradle_sha256,
            "gradle_distribution_url": _gradle_distribution_url(adapter.gradle),
            "data_pack_version": adapter.data_pack_version,
            "resource_pack_version": adapter.resource_pack_version,
            "resource_pack_format": adapter.resource_pack_format,
            "release_metadata_url": adapter.release_metadata_url,
            "source_api_family": adapter.source_api_family,
            "deterministic_module_kinds": sorted(adapter.deterministic_module_kinds),
        }
        payload["receipt_sha256"] = platform_receipt_sha256(payload)
        if extra:
            payload.update(extra)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    original_write_contract = generator.FabricProjectGenerator._write_contract

    @wraps(original_write_contract)
    def write_contract(self, root: Path, spec) -> None:
        original_write_contract(self, root, spec)
        lock = spec.platform
        if not _full_receipt_present(lock):
            raise SpecValidationError(
                "Generation requires the complete approval-bound execution platform receipt."
            )
        write_project_platform_lock(root, _adapter_from_receipt(lock))

    platform_resolver.lock_from_adapter = lock_from_adapter
    platform_catalog.adapter_for_lock_values = adapter_for_lock_values
    platform_catalog.adapter_from_project = adapter_from_project

    # Modules may have imported these functions before late reconciliation.
    generator.adapter_for_lock_values = adapter_for_lock_values
    generator.FabricProjectGenerator._write_contract = write_contract
    runner.adapter_from_project = adapter_from_project

    # Bootstrap-installed contracts imported these callables by value. Rebind their
    # module globals so every approved execution/validation path uses the frozen receipt.
    platform_generation_contract.adapter_for_lock_values = adapter_for_lock_values
    platform_runtime_contract.adapter_for_lock_values = adapter_for_lock_values
    platform_runtime_contract.adapter_from_project = adapter_from_project
    platform_live_execution_contract.adapter_for_lock_values = adapter_for_lock_values
    platform_live_execution_contract.adapter_from_project = adapter_from_project
    platform_validation_contract.adapter_for_lock_values = adapter_for_lock_values
    platform_validation_contract.adapter_from_project = adapter_from_project

    def generation_lock_writer(project_root: Path, adapter) -> None:
        write_project_platform_lock(project_root, adapter)

    platform_generation_contract._write_platform_lock = generation_lock_writer

    _INSTALLED = True


__all__ = ["install"]
