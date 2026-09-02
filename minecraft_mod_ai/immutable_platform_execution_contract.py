from __future__ import annotations

"""Freeze the complete executable platform receipt across approval and execution.

Resolution may use live provider evidence. Once an adapter is selected, however, every
coordinate needed by generation/build validation is copied into the approval-bound lock.
Downstream reconstruction uses only that receipt and therefore cannot silently drift when
Fabric/Minecraft metadata changes later.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_INSTALLED = False


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = str(value or "").split(".")
    if not parts or not all(part.isdigit() for part in parts):
        return ()
    return tuple(int(part) for part in parts)


def _receipt_value(value: Any, name: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _gradle_distribution_url(version: str) -> str:
    return f"https://services.gradle.org/distributions/gradle-{version}-bin.zip"


def _validate_cross_coordinate_consistency(lock: Any) -> None:
    from .spec import SpecValidationError

    minecraft = str(lock.minecraft_version)
    java = str(lock.java_version)
    yarn = str(lock.yarn_mappings)
    fabric_api = str(lock.fabric_api)

    mc_tuple = _version_tuple(minecraft)
    if mc_tuple and mc_tuple[0] == 1 and mc_tuple >= (1, 20, 5) and int(java) < 21:
        raise SpecValidationError(
            f"Minecraft {minecraft} requires a Java 21+ execution target; got Java {java}."
        )

    if yarn.casefold() not in {"mojang", "official", "official_mojang"}:
        match = re.match(r"^(\d+(?:\.\d+){1,2})\+", yarn)
        if match and match.group(1) != minecraft:
            raise SpecValidationError(
                "Platform lock mappings coordinate disagrees with minecraft_version."
            )

    api_match = re.search(r"\+(\d+(?:\.\d+){1,2})$", fabric_api)
    if api_match and api_match.group(1) != minecraft:
        raise SpecValidationError(
            "Platform lock Fabric API coordinate disagrees with minecraft_version."
        )


def _full_receipt_present(value: Any) -> bool:
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
    )
    if not all(
        _receipt_value(value, name, None) not in (None, "", 0)
        for name in required_nonempty
    ):
        return False
    # An empty deterministic-module list is meaningful for live targets, so require
    # field presence rather than truthiness.
    return _receipt_value(value, "deterministic_module_kinds", None) is not None


def _any_extended_receipt_field(value: Any) -> bool:
    return any(
        _receipt_value(value, name, None) not in (None, "", 0)
        for name in (
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
            "deterministic_module_kinds",
        )
    )


def _adapter_from_receipt(value: Any):
    from .platform_catalog import PlatformAdapter
    from .spec import SpecValidationError

    if not _full_receipt_present(value):
        raise SpecValidationError(
            "Execution platform receipt is incomplete; live provider re-resolution is forbidden."
        )

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
        spec,
    )
    from .spec import SpecValidationError

    LegacyPlatformLock = spec.PlatformLock

    @dataclass(frozen=True)
    class ExecutionPlatformLock(LegacyPlatformLock):
        adapter_id: str = ""
        mappings_kind: str = ""
        mappings_version: str = ""
        gradle_sha256: str = ""
        gradle_distribution_url: str = ""
        data_pack_version: str = ""
        resource_pack_version: str = ""
        resource_pack_format: int = 0
        release_metadata_url: str = ""
        source_api_family: str = ""
        deterministic_module_kinds: tuple[str, ...] = ()

        def validate(self) -> None:
            super().validate()
            _validate_cross_coordinate_consistency(self)

            if _any_extended_receipt_field(self) and not _full_receipt_present(self):
                raise SpecValidationError(
                    "Execution platform receipt is partial; all immutable provider fields are required."
                )
            if not _full_receipt_present(self):
                return

            if not re.fullmatch(r"[0-9a-f]{64}", self.gradle_sha256.casefold()):
                raise SpecValidationError("Platform lock gradle_sha256 must be a full SHA-256.")
            if self.gradle_distribution_url != _gradle_distribution_url(self.gradle):
                raise SpecValidationError(
                    "Platform lock Gradle distribution URL disagrees with the pinned version."
                )
            parsed = urlparse(self.gradle_distribution_url)
            if parsed.scheme != "https" or parsed.hostname != "services.gradle.org":
                raise SpecValidationError(
                    "Platform lock Gradle distribution URL must use the official HTTPS host."
                )
            if self.mappings_kind not in {"mojang", "yarn"}:
                raise SpecValidationError("Platform lock mappings_kind must be mojang or yarn.")
            if self.mappings_version != self.yarn_mappings:
                raise SpecValidationError(
                    "Platform lock mappings_version must equal the executable mappings coordinate."
                )
            if any(not str(item).strip() for item in self.deterministic_module_kinds):
                raise SpecValidationError(
                    "Platform lock deterministic_module_kinds contains an empty capability."
                )
            if len(set(self.deterministic_module_kinds)) != len(self.deterministic_module_kinds):
                raise SpecValidationError(
                    "Platform lock deterministic_module_kinds must not contain duplicates."
                )
            if type(self.resource_pack_format) is not int or self.resource_pack_format <= 0:
                raise SpecValidationError(
                    "Platform lock resource_pack_format must be a positive integer."
                )
            try:
                resource_major = int(self.resource_pack_version.split(".", 1)[0])
            except ValueError as exc:
                raise SpecValidationError(
                    "Platform lock resource_pack_version must start with a numeric major."
                ) from exc
            if resource_major != self.resource_pack_format:
                raise SpecValidationError(
                    "Platform lock resource pack format disagrees with resource_pack_version."
                )
            release = urlparse(self.release_metadata_url)
            if release.scheme != "https" or release.hostname not in {
                "www.minecraft.net",
                "feedback.minecraft.net",
                "piston-meta.mojang.com",
                "launcher.mojang.com",
            }:
                raise SpecValidationError(
                    "Platform lock release metadata must be an official Minecraft/Mojang HTTPS URL."
                )

    ExecutionPlatformLock.__name__ = "PlatformLock"
    ExecutionPlatformLock.__qualname__ = "PlatformLock"
    ExecutionPlatformLock.__module__ = spec.__name__

    def lock_from_adapter(adapter):
        adapter.validate()
        lock = ExecutionPlatformLock(
            edition=adapter.edition,
            loader=adapter.loader,
            minecraft_version=adapter.minecraft_version,
            java_version=adapter.java_version,
            yarn_mappings=adapter.yarn_mappings,
            fabric_loader=adapter.fabric_loader,
            fabric_api=adapter.fabric_api,
            fabric_loom=adapter.fabric_loom,
            gradle=adapter.gradle,
            adapter_id=adapter.adapter_id,
            mappings_kind=adapter.mappings_kind,
            mappings_version=adapter.mappings_version,
            gradle_sha256=adapter.gradle_sha256,
            gradle_distribution_url=_gradle_distribution_url(adapter.gradle),
            data_pack_version=adapter.data_pack_version,
            resource_pack_version=adapter.resource_pack_version,
            resource_pack_format=adapter.resource_pack_format,
            release_metadata_url=adapter.release_metadata_url,
            source_api_family=adapter.source_api_family,
            deterministic_module_kinds=tuple(sorted(adapter.deterministic_module_kinds)),
        )
        lock.validate()
        return lock

    original_adapter_from_project = platform_catalog.adapter_from_project

    def adapter_for_lock_values(value):
        # This function is an execution boundary.  Missing receipt fields are not
        # rediscovered from mutable provider state after approval.
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
        # approval.  Once MMM writes a lock, that lock is authoritative and fail-closed.
        return original_adapter_from_project(project_root)

    def write_project_platform_lock(project_root: Path, adapter, *, extra: dict[str, Any] | None = None) -> None:
        target = Path(project_root) / ".minecraft_ai" / "platform-lock.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "mmm/generated-platform-lock-v3",
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
        if extra:
            payload.update(extra)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    original_write_contract = generator.FabricProjectGenerator._write_contract

    from functools import wraps

    @wraps(original_write_contract)
    def write_contract(self, root: Path, spec_value) -> None:
        original_write_contract(self, root, spec_value)
        lock = spec_value.platform
        if not _full_receipt_present(lock):
            raise SpecValidationError(
                "Generation requires the complete approval-bound execution platform receipt."
            )
        write_project_platform_lock(root, _adapter_from_receipt(lock))

    spec.PlatformLock = ExecutionPlatformLock
    platform_resolver.PlatformLock = ExecutionPlatformLock
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
