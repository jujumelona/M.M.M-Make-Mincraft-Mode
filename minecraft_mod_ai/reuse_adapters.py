from __future__ import annotations

"""Deterministic source/resource adapters for reusable donor artifacts.

Mechanical source adaptation is deliberately separated from build-infrastructure
ownership.  Donor Gradle scripts, wrappers, and wrapper metadata are never adapted
or overlaid into a proof sandbox; the MMM verified scaffold remains authoritative.
Dependency coordinates are owned exclusively by dependency_resolver.py.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterReceipt:
    adapter_name: str
    applied: bool
    modified_files: tuple[str, ...]
    pre_hashes: Mapping[str, str]
    post_hashes: Mapping[str, str]
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "applied": self.applied,
            "modified_files": list(self.modified_files),
            "pre_hashes": dict(self.pre_hashes),
            "post_hashes": dict(self.post_hashes),
            "details": self.details,
        }


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def is_host_build_infrastructure(path: str) -> bool:
    """Return True for paths owned by the verified target build scaffold."""

    normalized = str(path or "").replace("\\", "/").strip("/")
    return normalized in {
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradle.properties",
        "gradlew",
        "gradlew.bat",
    } or normalized.startswith("gradle/wrapper/")


class PackageRelocationAdapter:
    """Relocate donor Java package declarations while preserving subpackages."""

    def can_apply(
        self,
        files: Mapping[str, str],
        target_context: Mapping[str, Any],
    ) -> bool:
        target_pkg = str(target_context.get("target_package") or "").strip()
        return bool(target_pkg) and any(path.endswith(".java") for path in files)

    def apply(
        self,
        files: dict[str, str],
        target_context: Mapping[str, Any],
    ) -> AdapterReceipt:
        target_pkg = str(
            target_context.get("target_package") or "ai.minecraft.generated.mod"
        ).strip()
        modified: list[str] = []
        pre_hashes: dict[str, str] = {}
        post_hashes: dict[str, str] = {}

        donor_packages: set[str] = set()
        for path, content in files.items():
            if not path.endswith(".java"):
                continue
            match = re.search(
                r"^\s*package\s+([A-Za-z0-9_.]+)\s*;",
                content,
                re.MULTILINE,
            )
            if match:
                donor_packages.add(match.group(1).strip())

        root_prefix = ""
        if donor_packages:
            split_packages = [package.split(".") for package in donor_packages]
            common_parts: list[str] = []
            for index in range(min(len(package) for package in split_packages)):
                part = split_packages[0][index]
                if all(package[index] == part for package in split_packages):
                    common_parts.append(part)
                else:
                    break
            root_prefix = ".".join(common_parts)

        def relocate_package(old_package: str) -> str:
            if root_prefix and old_package.startswith(root_prefix):
                suffix = old_package[len(root_prefix) :].strip(".")
                return f"{target_pkg}.{suffix}" if suffix else target_pkg
            return target_pkg

        for path in list(files):
            if not path.endswith(".java"):
                continue
            original = files[path]
            pre_hashes[path] = _sha(original)
            updated = original
            package_match = re.search(
                r"^\s*package\s+([A-Za-z0-9_.]+)\s*;",
                updated,
                re.MULTILINE,
            )
            if package_match:
                old_package = package_match.group(1).strip()
                new_package = relocate_package(old_package)
                updated = re.sub(
                    r"^\s*package\s+[A-Za-z0-9_.]+\s*;",
                    f"package {new_package};",
                    updated,
                    flags=re.MULTILINE,
                )

            for donor_package in sorted(donor_packages, key=lambda item: -len(item)):
                new_package = relocate_package(donor_package)
                if donor_package == new_package:
                    continue
                updated = re.sub(
                    rf"^\s*import\s+{re.escape(donor_package)}\.([A-Za-z0-9_*]+)\s*;",
                    rf"import {new_package}.\1;",
                    updated,
                    flags=re.MULTILINE,
                )

            if updated != original:
                files[path] = updated
                modified.append(path)
                post_hashes[path] = _sha(updated)

        return AdapterReceipt(
            adapter_name="PackageRelocationAdapter",
            applied=bool(modified),
            modified_files=tuple(modified),
            pre_hashes=pre_hashes,
            post_hashes=post_hashes,
            details=(
                f"Relocated {len(modified)} Java files to target root '{target_pkg}'."
            ),
        )


class ModIdRewriteAdapter:
    """Rewrite donor mod IDs and resource namespaces to the target mod ID."""

    def can_apply(
        self,
        files: Mapping[str, str],
        target_context: Mapping[str, Any],
    ) -> bool:
        target_modid = str(target_context.get("target_modid") or "").strip()
        donor_modid = str(target_context.get("donor_modid") or "").strip()
        return bool(target_modid) and (
            bool(donor_modid)
            or any("Identifier" in content or "mod.json" in path for path, content in files.items())
        )

    def apply(
        self,
        files: dict[str, str],
        target_context: Mapping[str, Any],
    ) -> AdapterReceipt:
        target_modid = str(
            target_context.get("target_modid") or "generated_mod"
        ).strip()
        donor_modid = str(target_context.get("donor_modid") or "").strip()
        modified: list[str] = []
        pre_hashes: dict[str, str] = {}
        post_hashes: dict[str, str] = {}

        if not donor_modid:
            for path in files:
                match = re.search(
                    r"src/main/resources/(?:assets|data)/([a-z0-9_.-]+)/",
                    path,
                )
                if match:
                    donor_modid = match.group(1).strip()
                    break

        for path in list(files):
            original = files[path]
            pre_hashes[path] = _sha(original)
            updated = original
            if donor_modid and donor_modid != target_modid:
                updated = updated.replace(
                    f'"{donor_modid}:',
                    f'"{target_modid}:',
                )
                updated = updated.replace(
                    f"'{donor_modid}:",
                    f"'{target_modid}:",
                )
                updated = re.sub(
                    rf'new\s+Identifier\s*\(\s*"{re.escape(donor_modid)}"\s*,\s*',
                    f'new Identifier("{target_modid}", ',
                    updated,
                )
                updated = re.sub(
                    rf'Identifier\.of\s*\(\s*"{re.escape(donor_modid)}"\s*,\s*',
                    f'Identifier.of("{target_modid}", ',
                    updated,
                )
                if path.endswith("fabric.mod.json"):
                    updated = re.sub(
                        rf'"id"\s*:\s*"{re.escape(donor_modid)}"',
                        f'"id": "{target_modid}"',
                        updated,
                    )
                    updated = re.sub(
                        rf'"{re.escape(donor_modid)}\.mixins\.json"',
                        f'"{target_modid}.mixins.json"',
                        updated,
                    )
                    updated = re.sub(
                        rf'"{re.escape(donor_modid)}\.accesswidener"',
                        f'"{target_modid}.accesswidener"',
                        updated,
                    )

            if updated != original:
                files[path] = updated
                modified.append(path)
                post_hashes[path] = _sha(updated)

        if donor_modid and donor_modid != target_modid:
            for old_path in list(files):
                for folder in ("assets", "data"):
                    target_prefix = (
                        f"src/main/resources/{folder}/{donor_modid}/"
                    )
                    new_prefix = f"src/main/resources/{folder}/{target_modid}/"
                    if old_path.startswith(target_prefix):
                        new_path = old_path.replace(target_prefix, new_prefix, 1)
                        content = files.pop(old_path)
                        files[new_path] = content
                        if new_path not in modified:
                            modified.append(new_path)
                        post_hashes[new_path] = _sha(content)

        return AdapterReceipt(
            adapter_name="ModIdRewriteAdapter",
            applied=bool(modified),
            modified_files=tuple(modified),
            pre_hashes=pre_hashes,
            post_hashes=post_hashes,
            details=(
                f"Rewrote mod ID from '{donor_modid}' to '{target_modid}' "
                f"across {len(modified)} files."
            ),
        )


class FabricApiMigrationAdapter:
    """Apply deterministic migrations for reviewed Fabric/Minecraft API renames."""

    def can_apply(
        self,
        files: Mapping[str, str],
        target_context: Mapping[str, Any],
    ) -> bool:
        del target_context
        return any(
            "FabricItemSettings" in content
            or "Registry.ITEM" in content
            or "Registry.BLOCK" in content
            or "Registry.ENTITY_TYPE" in content
            for content in files.values()
        )

    def apply(
        self,
        files: dict[str, str],
        target_context: Mapping[str, Any],
    ) -> AdapterReceipt:
        del target_context
        modified: list[str] = []
        pre_hashes: dict[str, str] = {}
        post_hashes: dict[str, str] = {}

        for path, original in list(files.items()):
            if not path.endswith(".java"):
                continue
            pre_hashes[path] = _sha(original)
            updated = original
            if "FabricItemSettings" in updated:
                updated = re.sub(
                    r"\bnew\s+FabricItemSettings\s*\(\s*\)",
                    "new Item.Settings()",
                    updated,
                )
                updated = re.sub(
                    r"import\s+net\.fabricmc\.fabric\.api\.item\.v1\.FabricItemSettings\s*;",
                    "",
                    updated,
                )

            replacements = {
                "Registry.ITEM": "Registries.ITEM",
                "Registry.BLOCK": "Registries.BLOCK",
                "Registry.ENTITY_TYPE": "Registries.ENTITY_TYPE",
            }
            if any(old in updated for old in replacements):
                for old, new in replacements.items():
                    updated = updated.replace(old, new)
                if "net.minecraft.registry.Registries" not in updated:
                    updated = "import net.minecraft.registry.Registries;\n" + updated

            if updated != original:
                files[path] = updated
                modified.append(path)
                post_hashes[path] = _sha(updated)

        return AdapterReceipt(
            adapter_name="FabricApiMigrationAdapter",
            applied=bool(modified),
            modified_files=tuple(modified),
            pre_hashes=pre_hashes,
            post_hashes=post_hashes,
            details=f"Migrated Fabric API conventions across {len(modified)} files.",
        )


class DependencyAdaptationPlan:
    """Compatibility facade; dependency_resolver remains the sole authority."""

    @classmethod
    def resolve_coordinate(
        cls,
        dep_name: str,
        loader: str,
        mc_version: str,
    ) -> tuple[str, str] | None:
        del cls
        from .dependency_resolver import resolve_dependency_for_target

        receipt = resolve_dependency_for_target(
            dep_name,
            target_loader=loader,
            target_minecraft=mc_version,
        )
        if not receipt.is_resolved:
            return None
        return receipt.repository, receipt.resolved_coordinate

    @classmethod
    def inject_dependencies_into_build_gradle(
        cls,
        build_gradle_content: str,
        required_dependencies: Sequence[str],
        *,
        loader: str = "fabric",
        minecraft_version: str = "1.21.1",
        is_kotlin_dsl: bool = False,
    ) -> tuple[str, bool]:
        del cls
        from .dependency_resolver import (
            inject_resolved_dependencies_into_build_gradle,
            resolve_dependency_for_target,
        )

        receipts = tuple(
            resolve_dependency_for_target(
                dependency,
                target_loader=loader,
                target_minecraft=minecraft_version,
            )
            for dependency in required_dependencies
        )
        return inject_resolved_dependencies_into_build_gradle(
            build_gradle_content,
            receipts,
            is_kotlin_dsl=is_kotlin_dsl,
        )


class ResidualSymbolAnalyzer:
    """Extract honest residual symbol gaps without fake stubs."""

    @classmethod
    def analyze_unresolved_symbols(
        cls,
        unresolved_symbols: Sequence[str],
    ) -> tuple[str, ...]:
        del cls
        return tuple(dict.fromkeys(symbol for symbol in unresolved_symbols if symbol))


def apply_deterministic_adapters(
    files: Mapping[str, str | bytes],
    target_context: Mapping[str, Any],
) -> tuple[dict[str, str | bytes], tuple[AdapterReceipt, ...]]:
    """Adapt reusable donor artifacts while excluding donor build infrastructure."""

    working_files: dict[str, str] = {}
    binary_files: dict[str, bytes] = {}
    excluded_paths: list[str] = []
    excluded_hashes: dict[str, str] = {}

    for path, content in files.items():
        if is_host_build_infrastructure(path):
            excluded_paths.append(path)
            excluded_hashes[path] = (
                _sha_bytes(content)
                if isinstance(content, bytes)
                else _sha(str(content))
            )
            continue
        if isinstance(content, bytes):
            try:
                working_files[path] = content.decode("utf-8")
            except UnicodeDecodeError:
                binary_files[path] = content
        else:
            working_files[path] = content

    receipts: list[AdapterReceipt] = []
    if excluded_paths:
        receipts.append(
            AdapterReceipt(
                adapter_name="HostBuildInfrastructureExclusion",
                applied=True,
                modified_files=tuple(sorted(excluded_paths)),
                pre_hashes=excluded_hashes,
                post_hashes={},
                details=(
                    "Excluded donor Gradle/build infrastructure so the verified MMM "
                    "target scaffold remains authoritative."
                ),
            )
        )

    adapters = (
        PackageRelocationAdapter(),
        ModIdRewriteAdapter(),
        FabricApiMigrationAdapter(),
    )
    for adapter in adapters:
        if adapter.can_apply(working_files, target_context):
            receipts.append(adapter.apply(working_files, target_context))

    result_files: dict[str, str | bytes] = {**working_files, **binary_files}
    return result_files, tuple(receipts)
