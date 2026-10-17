"""Install one byte-verified CurseForge Fabric dependency for local builds.

The remote JAR stays OUTSIDE the output project and is NEVER bundled into
released artifacts by this installer. The generated mod declares a Fabric
runtime dependency; downstream users must install the external mod separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def install_verified_fabric_dependency(
    project_root: str | Path,
    proof: Mapping[str, Any],
    *,
    minecraft_version: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if (proof.get("schema_version") != "mmm/curseforge-dependency-proof-v1"
            or proof.get("mode") != "dependency"
            or proof.get("minecraft_version") != minecraft_version
            or proof.get("loader") != "fabric"
            or proof.get("local_integrity_passed") is not True
            or proof.get("license_policy_confirmed") is not True
            or proof.get("redistribution_authorized") is not False):
        raise ValueError("VERIFIED_DEPENDENCY_PROOF_INVALID")
    jar = Path(str(proof.get("jar_path") or "")).expanduser().resolve()
    if not jar.is_file() or jar.is_relative_to(root):
        raise ValueError("VERIFIED_DEPENDENCY_JAR_NOT_EXTERNAL")
    expected = str(proof.get("sha256") or "").casefold()
    if len(expected) != 64 or hashlib.sha256(jar.read_bytes()).hexdigest() != expected:
        raise ValueError("VERIFIED_DEPENDENCY_SHA256_MISMATCH")
    mod_id = str(proof.get("fabric_mod_id") or "")
    if not mod_id or not mod_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("VERIFIED_DEPENDENCY_MOD_ID_INVALID")
    build = root / "build.gradle"
    descriptor = root / "src/main/resources/fabric.mod.json"
    if not build.is_file() or not descriptor.is_file():
        raise ValueError("VERIFIED_DEPENDENCY_FABRIC_PROJECT_REQUIRED")
    original = build.read_text(encoding="utf-8")
    info = json.loads(descriptor.read_text(encoding="utf-8"))
    if not isinstance(info, dict) or not isinstance(info.get("depends"), dict):
        raise ValueError("VERIFIED_DEPENDENCY_FABRIC_METADATA_INVALID")
    if mod_id in info["depends"] and info["depends"][mod_id] != "*":
        raise ValueError("VERIFIED_DEPENDENCY_CONFLICTING_RUNTIME_RANGE")
    # Gradle Groovy DSL single-quoted path, with escaping only for a host-bound
    # local path; this is NOT a download URL or a generated remote build script.
    encoded_path = str(jar).replace("\\", "\\\\").replace("'", "\\'")
    line = f"    modImplementation files('{encoded_path}')"
    marker = "// MMM:verified-curseforge-dependency:" + expected[:20]
    patch = "\n" + marker + "\ndependencies {\n" + line + "\n}\n"
    if marker in original:
        if patch.strip() not in original:
            raise ValueError("VERIFIED_DEPENDENCY_GRADLE_DRIFT")
        status = "ALREADY_INSTALLED"
        updated = original
    else:
        status = "INSTALLED"
        updated = original.rstrip() + "\n" + patch
    info["depends"][mod_id] = "*"
    updated_meta = json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    old_meta = descriptor.read_text(encoding="utf-8")
    try:
        build.write_text(updated, encoding="utf-8")
        descriptor.write_text(updated_meta, encoding="utf-8")
    except OSError:
        build.write_text(original, encoding="utf-8")
        descriptor.write_text(old_meta, encoding="utf-8")
        raise
    return {
        "schema_version": "mmm/verified-local-dependency-install-v1",
        "status": status, "fabric_mod_id": mod_id,
        "project_id": proof["project_id"], "file_id": proof["file_id"],
        "sha256": expected,
        "minecraft_version": minecraft_version, "loader": "fabric",
        "external_jar_location": str(jar),
        "redistributed": False,
        "target_compiled": False, "runtime_verified": False,
    }
