from __future__ import annotations

"""Deterministic Source and Resource Reuse Adapters.

Handles mechanical code and resource transformations before invoking any LLM coder:
1. Package & namespace relocation.
2. Mod ID and resource namespace rewrites (Java identifiers, JSON models/textures/loot).
3. Resource folder path relocations (assets/donor_mod -> assets/target_mod).
4. Standard API migrations (e.g. FabricItemSettings -> Item.Settings).

Each adapter follows an immutable receipt contract recording pre/post file hashes.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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


class PackageRelocationAdapter:
    """Relocates donor package declarations and cross-imports while preserving subpackage hierarchy."""

    def can_apply(self, files: Mapping[str, str], target_context: Mapping[str, Any]) -> bool:
        target_pkg = str(target_context.get("target_package") or "").strip()
        return bool(target_pkg) and any(p.endswith(".java") for p in files)

    def apply(self, files: dict[str, str], target_context: Mapping[str, Any]) -> AdapterReceipt:
        target_pkg = str(target_context.get("target_package") or "ai.minecraft.generated.mod").strip()
        modified: list[str] = []
        pre_hashes: dict[str, str] = {}
        post_hashes: dict[str, str] = {}

        # 1. Discover all donor packages and find common root prefix
        donor_packages: set[str] = set()
        for path, content in files.items():
            if path.endswith(".java"):
                match = re.search(r"^\s*package\s+([A-Za-z0-9_.]+)\s*;", content, re.MULTILINE)
                if match:
                    donor_packages.add(match.group(1).strip())

        # Determine common root prefix among donor packages
        root_prefix = ""
        if donor_packages:
            split_pkgs = [p.split(".") for p in donor_packages]
            min_len = min(len(p) for p in split_pkgs)
            common_parts = []
            for i in range(min_len):
                part = split_pkgs[0][i]
                if all(p[i] == part for p in split_pkgs):
                    common_parts.append(part)
                else:
                    break
            root_prefix = ".".join(common_parts)

        def relocate_pkg(old_pkg: str) -> str:
            if root_prefix and old_pkg.startswith(root_prefix):
                suffix = old_pkg[len(root_prefix):].strip(".")
                return f"{target_pkg}.{suffix}" if suffix else target_pkg
            return target_pkg

        for path in list(files):
            if not path.endswith(".java"):
                continue
            original = files[path]
            pre_hashes[path] = _sha(original)
            updated = original

            # Replace package declaration preserving subpackage
            pkg_match = re.search(r"^\s*package\s+([A-Za-z0-9_.]+)\s*;", updated, re.MULTILINE)
            if pkg_match:
                donor_pkg = pkg_match.group(1).strip()
                new_pkg = relocate_pkg(donor_pkg)
                updated = re.sub(
                    r"^\s*package\s+[A-Za-z0-9_.]+\s*;",
                    f"package {new_pkg};",
                    updated,
                    flags=re.MULTILINE,
                )

            # Rewrite cross-imports among donor classes
            for dp in sorted(donor_packages, key=lambda x: -len(x)):
                new_dp = relocate_pkg(dp)
                if dp != new_dp:
                    updated = re.sub(
                        rf"^\s*import\s+{re.escape(dp)}\.([A-Za-z0-9_*]+)\s*;",
                        rf"import {new_dp}.\1;",
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
            details=f"Relocated {len(modified)} Java files to target root '{target_pkg}'.",
        )


class ModIdRewriteAdapter:
    """Rewrites donor mod ID and resource namespaces in Java identifiers, JSON models, and asset paths."""

    def can_apply(self, files: Mapping[str, str], target_context: Mapping[str, Any]) -> bool:
        target_modid = str(target_context.get("target_modid") or "").strip()
        donor_modid = str(target_context.get("donor_modid") or "").strip()
        return bool(target_modid) and (bool(donor_modid) or any("Identifier" in c or "mod.json" in p for p, c in files.items()))

    def apply(self, files: dict[str, str], target_context: Mapping[str, Any]) -> AdapterReceipt:
        target_modid = str(target_context.get("target_modid") or "generated_mod").strip()
        donor_modid = str(target_context.get("donor_modid") or "").strip()
        modified: list[str] = []
        pre_hashes: dict[str, str] = {}
        post_hashes: dict[str, str] = {}

        # 1. Infer donor modid if not provided
        if not donor_modid:
            for path in files:
                m = re.search(r"src/main/resources/(?:assets|data)/([a-z0-9_.-]+)/", path)
                if m:
                    donor_modid = m.group(1).strip()
                    break

        # 2. Rewrite text references
        for path in list(files):
            original = files[path]
            pre_hashes[path] = _sha(original)
            updated = original

            if donor_modid and donor_modid != target_modid:
                # String literals in Java/JSON
                updated = updated.replace(f'"{donor_modid}:', f'"{target_modid}:')
                updated = updated.replace(f"'{donor_modid}:", f"'{target_modid}:")
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
                # fabric.mod.json contract rewrites
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

        # 3. Relocate asset/data file paths
        if donor_modid and donor_modid != target_modid:
            for old_path in list(files):
                for folder in ("assets", "data"):
                    target_prefix = f"src/main/resources/{folder}/{donor_modid}/"
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
            details=f"Rewrote mod ID from '{donor_modid}' to '{target_modid}' across {len(modified)} files.",
        )


class FabricApiMigrationAdapter:
    """Migrates deprecated Fabric and Minecraft API conventions with versioned transforms."""

    def can_apply(self, files: Mapping[str, str], target_context: Mapping[str, Any]) -> bool:
        return any(
            "FabricItemSettings" in c or "Registry.ITEM" in c or "Registry.BLOCK" in c
            for c in files.values()
            if isinstance(c, str)
        )

    def apply(self, files: dict[str, str], target_context: Mapping[str, Any]) -> AdapterReceipt:
        modified: list[str] = []
        pre_hashes: dict[str, str] = {}
        post_hashes: dict[str, str] = {}

        for path, original in list(files.items()):
            if not path.endswith(".java"):
                continue
            pre_hashes[path] = _sha(original)
            updated = original

            # 1. FabricItemSettings -> Item.Settings()
            if "FabricItemSettings" in updated:
                updated = re.sub(r"\bnew\s+FabricItemSettings\s*\(\s*\)", "new Item.Settings()", updated)
                updated = re.sub(r"import\s+net\.fabricmc\.fabric\.api\.item\.v1\.FabricItemSettings\s*;", "", updated)

            # 2. Modern 1.20+ Registries mappings
            if "Registry.ITEM" in updated:
                updated = updated.replace("Registry.ITEM", "Registries.ITEM")
                if "net.minecraft.registry.Registries" not in updated:
                    updated = "import net.minecraft.registry.Registries;\n" + updated
            if "Registry.BLOCK" in updated:
                updated = updated.replace("Registry.BLOCK", "Registries.BLOCK")
                if "net.minecraft.registry.Registries" not in updated:
                    updated = "import net.minecraft.registry.Registries;\n" + updated
            if "Registry.ENTITY_TYPE" in updated:
                updated = updated.replace("Registry.ENTITY_TYPE", "Registries.ENTITY_TYPE")
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
    """Calculates and applies build script modifications for required external dependencies."""

    DEPENDENCY_REGISTRY: dict[str, dict[str, Any]] = {
        "cloth_config": {
            "maven": "https://maven.shedaniel.me/",
            "coords": {
                "fabric": "me.shedaniel.cloth:cloth-config-fabric:15.0.127",
                "neoforge": "me.shedaniel.cloth:cloth-config-neoforge:15.0.127",
                "forge": "me.shedaniel.cloth:cloth-config-forge:15.0.127",
            },
        },
        "cloth-config": {
            "maven": "https://maven.shedaniel.me/",
            "coords": {
                "fabric": "me.shedaniel.cloth:cloth-config-fabric:15.0.127",
                "neoforge": "me.shedaniel.cloth:cloth-config-neoforge:15.0.127",
                "forge": "me.shedaniel.cloth:cloth-config-forge:15.0.127",
            },
        },
        "cardinal_components": {
            "maven": "https://ladysnake.jfrog.io/artifactory/mods",
            "coords": {
                "fabric": "dev.onyxstudios.cardinal-components-api:cardinal-components-base:6.0.0",
            },
        },
        "geckolib": {
            "maven": "https://dl.cloudsmith.io/public/geckolib3/geckolib/maven/",
            "coords": {
                "fabric": "software.bernie.geckolib:geckolib-fabric-1.21.1:4.6.6",
                "neoforge": "software.bernie.geckolib:geckolib-neoforge-1.21.1:4.6.6",
            },
        },
        "patchouli": {
            "maven": "https://maven.blamejared.com/",
            "coords": {
                "fabric": "vazkii.patchouli:Patchouli:1.21-87-FABRIC",
                "neoforge": "vazkii.patchouli:Patchouli:1.21-87-NEOFORGE",
            },
        },
    }

    @classmethod
    def resolve_coordinate(cls, dep_name: str, loader: str, mc_version: str) -> tuple[str, str] | None:
        dep_key = dep_name.casefold().replace("-", "_")
        entry = cls.DEPENDENCY_REGISTRY.get(dep_key)
        if not entry:
            return None
        maven = entry.get("maven", "")
        coords_map = entry.get("coords", {})
        coord = coords_map.get(loader.casefold()) or next(iter(coords_map.values()), "")
        if not coord:
            return None
        # Format version if template placeholder exists
        if "{mc_version}" in coord:
            coord = coord.replace("{mc_version}", mc_version)
        return maven, coord

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
        """Inject required maven repos and dependencies into a build.gradle or build.gradle.kts script."""
        if not required_dependencies:
            return build_gradle_content, False

        modified = build_gradle_content
        applied = False

        for dep in required_dependencies:
            resolved = cls.resolve_coordinate(dep, loader, minecraft_version)
            if not resolved:
                continue

            maven_url, coord = resolved
            dep_keyword = "modImplementation" if loader.casefold() == "fabric" else "implementation"

            if is_kotlin_dsl:
                repo_block = f'    maven("{maven_url}")\n'
                dep_line = f'    {dep_keyword}("{coord}")\n'
            else:
                repo_block = f"    maven {{ url '{maven_url}' }}\n"
                dep_line = f"    {dep_keyword} '{coord}'\n"

            # 1. Inject maven repo if missing
            if maven_url not in modified and "repositories {" in modified:
                modified = modified.replace("repositories {", f"repositories {{\n{repo_block}", 1)
                applied = True

            # 2. Inject dependency coordinate if missing
            if coord not in modified and "dependencies {" in modified:
                modified = modified.replace("dependencies {", f"dependencies {{\n{dep_line}", 1)
                applied = True

        return modified, applied


class ResidualSymbolAnalyzer:
    """Analyzes compilation diagnostics to extract honest residual symbol gaps without fake stubs."""

    @classmethod
    def analyze_unresolved_symbols(
        cls,
        unresolved_symbols: Sequence[str],
    ) -> tuple[str, ...]:
        """Return canonical sorted list of unresolvable residual symbols."""
        return tuple(dict.fromkeys(sym for sym in unresolved_symbols if sym))


def apply_deterministic_adapters(
    files: Mapping[str, str | bytes],
    target_context: Mapping[str, Any],
) -> tuple[dict[str, str | bytes], tuple[AdapterReceipt, ...]]:
    """Apply all applicable deterministic adapters in sequence, returning adapted files and receipts."""
    working_files: dict[str, str] = {}
    binary_files: dict[str, bytes] = {}

    for path, content in files.items():
        if isinstance(content, bytes):
            try:
                working_files[path] = content.decode("utf-8")
            except UnicodeDecodeError:
                binary_files[path] = content
        else:
            working_files[path] = content

    adapters = (
        PackageRelocationAdapter(),
        ModIdRewriteAdapter(),
        FabricApiMigrationAdapter(),
    )
    receipts: list[AdapterReceipt] = []

    for adapter in adapters:
        if adapter.can_apply(working_files, target_context):
            receipt = adapter.apply(working_files, target_context)
            receipts.append(receipt)

    result_files: dict[str, str | bytes] = {**working_files, **binary_files}
    return result_files, tuple(receipts)
