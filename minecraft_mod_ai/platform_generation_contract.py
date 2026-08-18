from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_for_lock_values


def install(generator_module: Any) -> None:
    """Bind generated artifacts to the exact provider receipt without target-era rewrites."""

    generator = generator_module.FabricProjectGenerator
    if getattr(generator.generate, "_mmm_dynamic_platform_generation", False):
        return

    original_generate = generator.generate

    @wraps(original_generate)
    def generate(self: Any, spec: Any, root: Path):
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


def _write_platform_lock(project_root: Path, adapter: Any) -> None:
    target = project_root / ".minecraft_ai" / "platform-lock.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mmm/generated-platform-lock-v1",
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
        "resource_pack_format": adapter.resource_pack_format,
        "source_api_family": adapter.source_api_family,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
