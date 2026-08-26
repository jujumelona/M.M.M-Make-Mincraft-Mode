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
from typing import Any, Mapping


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
    """Relocates donor package declarations and cross-imports to target project package namespace."""

    def can_apply(self, files: Mapping[str, str], target_context: Mapping[str, Any]) -> bool:
        target_pkg = str(target_context.get("target_package") or "").strip()
        return bool(target_pkg) and any(p.endswith(".java") for p in files)

    def apply(self, files: dict[str, str], target_context: Mapping[str, Any]) -> AdapterReceipt:
        target_pkg = str(target_context.get("target_package") or "ai.minecraft.generated.mod").strip()
        modified: list[str] = []
        pre_hashes: dict[str, str] = {}
        post_hashes: dict[str, str] = {}

        # Discover all donor packages
        donor_packages: set[str] = set()
        for path, content in files.items():
            if path.endswith(".java"):
                match = re.search(r"^\s*package\s+([A-Za-z0-9_.]+)\s*;", content, re.MULTILINE)
                if match:
                    donor_packages.add(match.group(1).strip())

        for path in list(files):
            if not path.endswith(".java"):
                continue
            original = files[path]
            pre_hashes[path] = _sha(original)
            updated = original

            # Replace package declaration
            pkg_match = re.search(r"^\s*package\s+([A-Za-z0-9_.]+)\s*;", updated, re.MULTILINE)
            if pkg_match:
                updated = re.sub(
                    r"^\s*package\s+[A-Za-z0-9_.]+\s*;",
                    f"package {target_pkg};",
                    updated,
                    flags=re.MULTILINE,
                )

            # Rewrite cross-imports among donor classes to target package
            for dp in donor_packages:
                if dp != target_pkg:
                    updated = re.sub(
                        rf"^\s*import\s+{re.escape(dp)}\.([A-Za-z0-9_*]+)\s*;",
                        rf"import {target_pkg}.\1;",
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
            details=f"Relocated {len(modified)} Java files to target package '{target_pkg}'.",
        )


class ModIdRewriteAdapter:
    """Rewrites donor mod ID and resource namespaces in Java identifiers, JSON models, and asset paths."""

    def can_apply(self, files: Mapping[str, str], target_context: Mapping[str, Any]) -> bool:
        target_modid = str(target_context.get("target_modid") or "").strip()
        donor_modid = str(target_context.get("donor_modid") or "").strip()
        return bool(target_modid) and (bool(donor_modid) or any("Identifier" in c for c in files.values()))

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
    """Migrates deprecated Fabric API conventions (e.g. FabricItemSettings -> Item.Settings)."""

    def can_apply(self, files: Mapping[str, str], target_context: Mapping[str, Any]) -> bool:
        return any("FabricItemSettings" in c for c in files.values() if isinstance(c, str))

    def apply(self, files: dict[str, str], target_context: Mapping[str, Any]) -> AdapterReceipt:
        modified: list[str] = []
        pre_hashes: dict[str, str] = {}
        post_hashes: dict[str, str] = {}

        for path, original in list(files.items()):
            if not path.endswith(".java"):
                continue
            pre_hashes[path] = _sha(original)
            updated = original

            if "FabricItemSettings" in updated:
                updated = re.sub(r"\bnew\s+FabricItemSettings\s*\(\s*\)", "new Item.Settings()", updated)
                updated = re.sub(r"import\s+net\.fabricmc\.fabric\.api\.item\.v1\.FabricItemSettings\s*;", "", updated)
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
