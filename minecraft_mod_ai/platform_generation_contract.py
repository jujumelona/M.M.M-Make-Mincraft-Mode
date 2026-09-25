from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_for_lock_values
from .spec import platform_receipt_sha256


def install(generator_module: Any) -> None:
    """Bind generated artifacts to the exact provider receipt without target-era rewrites."""

    generator = generator_module.FabricProjectGenerator
    if getattr(generator.generate, "_mmm_dynamic_platform_generation", False):
        return

    original_generate = generator.generate

    @wraps(original_generate)
    def generate(self: Any, spec: Any, root: Path):
        reviewed_kinds = tuple(getattr(spec.platform, "deterministic_module_kinds", ()) or ())
        if not reviewed_kinds:
            raise generator_module.GenerationError(
                f"Target {spec.platform.minecraft_version} has no reviewed deterministic module templates."
            )
        adapter = adapter_for_lock_values(spec.platform)
        result = original_generate(self, spec, root)
        project_root = Path(result.root).resolve()
        _write_platform_lock(project_root, adapter)
        _rewrite_gradle_properties(project_root, adapter)
        _rewrite_pack_metadata(project_root, spec.mod_name, adapter.resource_pack_format)
        return result

    generate._mmm_dynamic_platform_generation = True
    generate.__wrapped__ = original_generate
    generator.generate = generate


def _matching_existing_bootstrap(
    target: Path,
    adapter: Any,
) -> dict[str, Any] | None:
    """Preserve bootstrap evidence only when the existing lock is the same target."""

    if not target.is_file() or target.is_symlink():
        return None
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(existing, dict):
        return None
    expected = {
        "adapter_id": adapter.adapter_id,
        "loader": adapter.loader,
        "minecraft_version": adapter.minecraft_version,
        "java_version": adapter.java_version,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "fabric_loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
        "gradle_sha256": adapter.gradle_sha256,
    }
    if any(
        str(existing.get(key) or "") != str(value)
        for key, value in expected.items()
    ):
        return None
    bootstrap = existing.get("bootstrap")
    return dict(bootstrap) if isinstance(bootstrap, dict) else None


def _platform_context_fields(adapter: Any) -> dict[str, Any]:
    if not adapter.host_facts_json:
        return {}
    context = adapter.version_context
    return {
        "resolved_version_context": context.to_dict(),
        "context_id": context.context_id,
    }


def write_platform_lock(
    project_root: Path,
    adapter: Any,
    *,
    bootstrap: dict[str, Any] | None = None,
) -> None:
    """Persist the one complete approval/execution platform lock schema."""

    target = project_root / ".minecraft_ai" / "platform-lock.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_bootstrap = (
        dict(bootstrap)
        if bootstrap is not None
        else _matching_existing_bootstrap(target, adapter)
    )
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
        "host_facts_json": adapter.host_facts_json,
    }
    payload.update(_platform_context_fields(adapter))
    payload["receipt_sha256"] = platform_receipt_sha256(payload)
    if resolved_bootstrap is not None:
        payload["bootstrap"] = resolved_bootstrap
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_platform_lock(
    project_root: Path,
    adapter: Any,
    receipt: dict[str, Any] | None = None,
) -> None:
    """Compatibility entrypoint backed by the canonical complete writer."""

    write_platform_lock(project_root, adapter, bootstrap=receipt)

def _rewrite_gradle_properties(project_root: Path, adapter: Any) -> None:
    path = project_root / "gradle.properties"
    text = path.read_text(encoding="utf-8")
    additions = {
        "loader": adapter.loader,
        "java_version": adapter.java_version,
        "gradle_version": adapter.gradle,
        "platform_adapter": adapter.adapter_id,
    }
    lines = text.splitlines()
    existing = {
        line.split("=", 1)[0].strip()
        for line in lines
        if "=" in line and not line.lstrip().startswith("#")
    }
    for key, value in additions.items():
        if key not in existing:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _rewrite_pack_metadata(project_root: Path, mod_name: str, pack_format: int) -> None:
    path = project_root / "src" / "main" / "resources" / "pack.mcmeta"
    payload = {
        "pack": {
            "pack_format": int(pack_format),
            "description": f"{mod_name} resources",
        }
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
