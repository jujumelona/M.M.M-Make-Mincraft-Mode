from __future__ import annotations

"""Rebind Fabric's mutable official scaffold to the already-approved immutable target.

The official CLI is intentionally kept as the structural scaffold authority.  Its current
stable Loader/API/Loom/Gradle defaults are *not* execution authority after approval.  This
contract rewrites only version-bearing build metadata back to the approved PlatformAdapter,
then verifies the materialized project before execution continues.
"""

import json
import re
from pathlib import Path
from typing import Any

from .spec import platform_receipt_sha256

_INSTALLED = False


def _replace_property(path: Path, replacements: dict[str, str]) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Fabric template omitted required {path.name}.")
    lines = path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in raw:
            key = raw.split("=", 1)[0].strip()
            if key in replacements:
                output.append(f"{key}={replacements[key]}")
                seen.add(key)
                continue
        output.append(raw)
    for key, value in replacements.items():
        if key not in seen:
            output.append(f"{key}={value}")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _replace_exact_tokens(path: Path, replacements: dict[str, str]) -> None:
    if not path.is_file() or path.is_symlink():
        return
    text = path.read_text(encoding="utf-8")
    rewritten = text
    for old, new in replacements.items():
        if old and new and old != new:
            rewritten = rewritten.replace(old, new)
    if rewritten != text:
        path.write_text(rewritten, encoding="utf-8")


def _rewrite_java_release(path: Path, java_version: str) -> None:
    if not path.is_file() or path.is_symlink():
        return
    text = path.read_text(encoding="utf-8")
    patterns = (
        (r"(options\.release\s*=\s*)\d+", rf"\g<1>{java_version}"),
        (r"(options\.release\.set\(\s*)\d+(\s*\))", rf"\g<1>{java_version}\g<2>"),
        (r"JavaVersion\.VERSION_\d+", f"JavaVersion.VERSION_{java_version}"),
        (r"(JavaLanguageVersion\.of\(\s*)\d+(\s*\))", rf"\g<1>{java_version}\g<2>"),
    )
    rewritten = text
    for pattern, replacement in patterns:
        rewritten = re.sub(pattern, replacement, rewritten)
    if rewritten != text:
        path.write_text(rewritten, encoding="utf-8")


def _rewrite_loom_literal(path: Path, loom_version: str) -> None:
    if not path.is_file() or path.is_symlink():
        return
    text = path.read_text(encoding="utf-8")
    patterns = (
        r"(id\s+['\"]fabric-loom['\"]\s+version\s+['\"])([^'\"]+)(['\"])",
        r"(id\(\s*['\"]fabric-loom['\"]\s*\)\s*version\s*['\"])([^'\"]+)(['\"])",
    )
    rewritten = text
    for pattern in patterns:
        rewritten = re.sub(pattern, rf"\g<1>{loom_version}\g<3>", rewritten)
    if rewritten != text:
        path.write_text(rewritten, encoding="utf-8")


def _rewrite_wrapper(path: Path, adapter: Any) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Fabric template omitted the Gradle wrapper properties.")
    text = path.read_text(encoding="utf-8")
    rewritten, count = re.subn(
        r"gradle-[0-9][0-9A-Za-z_.-]*-bin\.zip",
        f"gradle-{adapter.gradle}-bin.zip",
        text,
    )
    if count != 1:
        raise RuntimeError("Could not deterministically rebind the generated Gradle wrapper.")
    sha_line = f"distributionSha256Sum={adapter.gradle_sha256}"
    if re.search(r"(?m)^distributionSha256Sum=.*$", rewritten):
        rewritten = re.sub(
            r"(?m)^distributionSha256Sum=.*$",
            sha_line,
            rewritten,
        )
    else:
        if not rewritten.endswith("\n"):
            rewritten += "\n"
        rewritten += sha_line + "\n"
    path.write_text(rewritten, encoding="utf-8")


def _literal_dependency_versions(root: Path) -> dict[str, set[str]]:
    found = {"loader": set(), "api": set(), "loom": set()}
    for name in ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(
            r"net\.fabricmc:fabric-loader:([0-9A-Za-z_.+\-]+)", text
        ):
            value = match.group(1)
            if "$" not in value:
                found["loader"].add(value)
        for match in re.finditer(
            r"net\.fabricmc\.fabric-api:fabric-api:([0-9A-Za-z_.+\-]+)", text
        ):
            value = match.group(1)
            if "$" not in value:
                found["api"].add(value)
        for pattern in (
            r"id\s+['\"]fabric-loom['\"]\s+version\s+['\"]([^'\"]+)['\"]",
            r"id\(\s*['\"]fabric-loom['\"]\s*\)\s*version\s*['\"]([^'\"]+)['\"]",
        ):
            for match in re.finditer(pattern, text):
                value = match.group(1)
                if "$" not in value:
                    found["loom"].add(value)
    return found


def _write_full_platform_lock(root: Path, adapter: Any, receipt: dict[str, Any]) -> None:
    """Persist the complete approval-bound receipt after scaffold rebind.

    This is the final writer in the live Fabric bootstrap path. It must therefore emit
    every field required by the offline execution lock contract, including the canonical
    receipt digest. Writing a legacy v2/v3 lock here would overwrite a complete lock from
    an earlier boundary and make the immediately following adapter_from_project() fail.
    """

    target = root / ".minecraft_ai" / "platform-lock.json"
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
        "gradle_distribution_url": (
            f"https://services.gradle.org/distributions/gradle-{adapter.gradle}-bin.zip"
        ),
        "data_pack_version": adapter.data_pack_version,
        "resource_pack_version": adapter.resource_pack_version,
        "resource_pack_format": adapter.resource_pack_format,
        "release_metadata_url": adapter.release_metadata_url,
        "source_api_family": adapter.source_api_family,
        "deterministic_module_kinds": sorted(adapter.deterministic_module_kinds),
    }
    payload["receipt_sha256"] = platform_receipt_sha256(payload)
    payload["bootstrap"] = dict(receipt)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rebind_scaffold(provider: Any, root: Path, adapter: Any) -> dict[str, str]:
    root = Path(root).resolve()
    properties_path = root / "gradle.properties"
    before = provider._read_properties(properties_path)
    actual_mc = before.get("minecraft_version", "")
    if actual_mc != adapter.minecraft_version:
        raise provider.FabricTemplateProviderError(
            "Fabric official template generated a different Minecraft target: "
            f"expected={adapter.minecraft_version}, actual={actual_mc!r}"
        )

    old_loader = before.get("loader_version", "")
    old_api = before.get("fabric_version", "") or before.get("fabric_api_version", "")
    old_loom = before.get("loom_version", "")

    replacements = {
        "minecraft_version": str(adapter.minecraft_version),
        "loader_version": str(adapter.fabric_loader),
        "loom_version": str(adapter.fabric_loom),
        "java_version": str(adapter.java_version),
    }
    if "fabric_api_version" in before and "fabric_version" not in before:
        replacements["fabric_api_version"] = str(adapter.fabric_api)
    else:
        replacements["fabric_version"] = str(adapter.fabric_api)
    if adapter.mappings_kind == "yarn":
        replacements["yarn_mappings"] = str(adapter.mappings_version)
    _replace_property(properties_path, replacements)

    exact_replacements = {
        old_loader: str(adapter.fabric_loader),
        old_api: str(adapter.fabric_api),
        old_loom: str(adapter.fabric_loom),
    }
    for name in ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"):
        path = root / name
        _replace_exact_tokens(path, exact_replacements)
        _rewrite_loom_literal(path, str(adapter.fabric_loom))
        _rewrite_java_release(path, str(adapter.java_version))

    _rewrite_wrapper(root / "gradle/wrapper/gradle-wrapper.properties", adapter)

    after = provider._read_properties(properties_path)
    actual_loader = after.get("loader_version", "")
    actual_api = after.get("fabric_version", "") or after.get("fabric_api_version", "")
    actual_loom = after.get("loom_version", "")
    actual_gradle = provider._gradle_wrapper_version(root)
    actual_java = provider._java_release(root)

    expected = {
        "minecraft_version": str(adapter.minecraft_version),
        "loader_version": str(adapter.fabric_loader),
        "fabric_api": str(adapter.fabric_api),
        "loom": str(adapter.fabric_loom),
        "gradle": str(adapter.gradle),
        "java": str(adapter.java_version),
    }
    actual = {
        "minecraft_version": after.get("minecraft_version", ""),
        "loader_version": actual_loader,
        "fabric_api": actual_api,
        "loom": actual_loom,
        "gradle": actual_gradle,
        "java": actual_java,
    }
    mismatches = [key for key in expected if expected[key] != actual[key]]
    if mismatches:
        raise provider.FabricTemplateProviderError(
            "Fabric official scaffold could not be rebound to the approved immutable target: "
            + ", ".join(
                f"{key} expected={expected[key]!r} actual={actual[key]!r}"
                for key in mismatches
            )
        )

    literal = _literal_dependency_versions(root)
    unexpected = {
        "loader": sorted(value for value in literal["loader"] if value != str(adapter.fabric_loader)),
        "api": sorted(value for value in literal["api"] if value != str(adapter.fabric_api)),
        "loom": sorted(value for value in literal["loom"] if value != str(adapter.fabric_loom)),
    }
    unexpected = {key: value for key, value in unexpected.items() if value}
    if unexpected:
        raise provider.FabricTemplateProviderError(
            "Fabric official scaffold retained version literals outside the approved receipt: "
            + json.dumps(unexpected, sort_keys=True)
        )
    return actual


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import fabric_official_template_provider as provider
    from . import platform_live_execution_contract as live_execution

    original = provider.bootstrap_fabric_project

    def bootstrap_fabric_project(
        *,
        project_root: str | Path,
        spec: Any,
        adapter: Any,
        cache_root: str | Path,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        scaffold_error: Exception | None = None
        receipt: dict[str, Any] | None = None
        try:
            receipt = original(
                project_root=root,
                spec=spec,
                adapter=adapter,
                cache_root=cache_root,
            )
        except provider.FabricTemplateProviderError as exc:
            scaffold_error = exc
            # Recover only after the official CLI actually produced the exact approved
            # Minecraft scaffold. CLI/download/path failures remain terminal.
            if not root.is_dir() or not (root / "gradle.properties").is_file():
                raise
            before = provider._read_properties(root / "gradle.properties")
            if before.get("minecraft_version", "") != str(adapter.minecraft_version):
                raise

        before = provider._read_properties(root / "gradle.properties")
        emitted = {
            "minecraft_version": before.get("minecraft_version", ""),
            "loader_version": before.get("loader_version", ""),
            "fabric_api": before.get("fabric_version", "") or before.get("fabric_api_version", ""),
            "loom": before.get("loom_version", ""),
            "gradle": provider._gradle_wrapper_version(root),
            "java": provider._java_release(root),
        }
        verified = _rebind_scaffold(provider, root, adapter)

        if receipt is None:
            deno = provider._ensure_deno(Path(cache_root).expanduser().resolve())
            receipt = {
                "schema_version": "mmm/fabric-official-template-v3",
                "provider": "fabricmc.net/cli",
                "provider_url": provider._FABRIC_CLI,
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "loader_version": adapter.fabric_loader,
                "fabric_api": adapter.fabric_api,
                "loom": adapter.fabric_loom,
                "gradle": adapter.gradle,
                "java": adapter.java_version,
                "mappings": adapter.mappings_kind,
                "deno": provider._deno_version(deno),
                "scaffold_rebind_reason": str(scaffold_error or ""),
            }
        else:
            receipt = dict(receipt)
            receipt["schema_version"] = "mmm/fabric-official-template-v3"
        receipt["scaffold_emitted_toolchain"] = emitted
        receipt["verified_generated_toolchain"] = verified
        receipt["approval_rebind"] = "EXACT"
        receipt["project_manifest_sha256"] = provider._manifest_hash(root)
        _write_full_platform_lock(root, adapter, receipt)
        return receipt

    provider.bootstrap_fabric_project = bootstrap_fabric_project
    # The live execution contract imported the callable by value during runtime bootstrap.
    live_execution.bootstrap_fabric_project = bootstrap_fabric_project
    live_execution._write_approved_target_lock = (
        lambda metadata, adapter, receipt: _write_full_platform_lock(
            Path(metadata).parent, adapter, receipt
        )
    )

    _INSTALLED = True


__all__ = ["_rebind_scaffold", "_write_full_platform_lock", "install"]
