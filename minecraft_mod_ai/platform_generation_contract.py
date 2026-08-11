from __future__ import annotations

import json
import shutil
from functools import wraps
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_for_lock_values


def install(generator_module: Any) -> None:
    generator = generator_module.FabricProjectGenerator
    if getattr(generator.generate, "_mmm_dynamic_platform_generation", False):
        return

    original_generate = generator.generate
    original_build_gradle = generator._build_gradle
    original_main_java = generator._main_java
    original_gametest_java = generator._gametest_java
    original_write_item = generator._write_item
    original_write_block = generator._write_block

    @wraps(original_build_gradle)
    def build_gradle(self: Any, spec: Any) -> str:
        adapter = adapter_for_lock_values(spec.platform)
        text = original_build_gradle(self, spec)
        text = text.replace("options.release = 17", f"options.release = {adapter.java_version}")
        text = text.replace(
            "sourceCompatibility = JavaVersion.VERSION_17",
            f"sourceCompatibility = JavaVersion.VERSION_{adapter.java_version}",
        )
        text = text.replace(
            "targetCompatibility = JavaVersion.VERSION_17",
            f"targetCompatibility = JavaVersion.VERSION_{adapter.java_version}",
        )
        if adapter.source_api_family == "fabric_1211":
            text = text.replace("/recipes/", "/recipe/")
            text = text.replace("/loot_tables/", "/loot_table/")
            text = text.replace("/advancements/", "/advancement/")
            text = text.replace("/tags/blocks/", "/tags/block/")
            text = text.replace("/tags/items/", "/tags/item/")
            text = text.replace("/tags/entity_types/", "/tags/entity_type/")
        return text

    @wraps(original_main_java)
    def main_java(self: Any, spec: Any, main_class: str) -> str:
        adapter = adapter_for_lock_values(spec.platform)
        if adapter.source_api_family != "fabric_1211":
            return original_main_java(self, spec, main_class)
        if spec.boss is not None:
            raise generator_module.GenerationError(
                "Minecraft 1.21.1 boss bootstrap is not a reviewed source adapter yet."
            )
        text = original_main_java(self, spec, main_class)
        text = text.replace(
            "import net.fabricmc.fabric.api.item.v1.FabricItemSettings;\n", ""
        )
        text = text.replace(
            "import net.fabricmc.fabric.api.object.builder.v1.block.FabricBlockSettings;\n",
            "import net.minecraft.block.AbstractBlock;\n",
        )
        text = text.replace("new FabricItemSettings()", "new Item.Settings()")
        text = text.replace(
            "FabricBlockSettings.copyOf(Blocks.STONE)",
            "AbstractBlock.Settings.copy(Blocks.STONE)",
        )
        text = text.replace("new Identifier(MOD_ID, name)", "Identifier.of(MOD_ID, name)")
        return text

    @wraps(original_gametest_java)
    def gametest_java(self: Any, spec: Any, main_class: str) -> str:
        adapter = adapter_for_lock_values(spec.platform)
        text = original_gametest_java(self, spec, main_class)
        if adapter.source_api_family == "fabric_1211":
            text = text.replace("new Identifier(", "Identifier.of(")
        return text

    @wraps(original_write_item)
    def write_item(self: Any, root: Path, spec: Any, content: Any) -> None:
        original_write_item(self, root, spec, content)
        adapter = adapter_for_lock_values(spec.platform)
        if adapter.source_api_family == "fabric_1211":
            _migrate_1211_data_tree(Path(root), spec.mod_id)

    @wraps(original_write_block)
    def write_block(self: Any, root: Path, spec: Any, content: Any) -> None:
        original_write_block(self, root, spec, content)
        adapter = adapter_for_lock_values(spec.platform)
        if adapter.source_api_family == "fabric_1211":
            _migrate_1211_data_tree(Path(root), spec.mod_id)

    @wraps(original_generate)
    def generate(self: Any, spec: Any, root: Path):
        adapter = adapter_for_lock_values(spec.platform)
        result = original_generate(self, spec, root)
        project_root = Path(result.root).resolve()
        if adapter.source_api_family == "fabric_1211":
            _migrate_1211_data_tree(project_root, spec.mod_id)
        _write_platform_lock(project_root, adapter)
        _rewrite_gradle_properties(project_root, adapter)
        _rewrite_pack_metadata(project_root, spec.mod_name, adapter.resource_pack_format)
        return result

    build_gradle._mmm_dynamic_platform_generation = True
    main_java._mmm_dynamic_platform_generation = True
    gametest_java._mmm_dynamic_platform_generation = True
    write_item._mmm_dynamic_platform_generation = True
    write_block._mmm_dynamic_platform_generation = True
    generate._mmm_dynamic_platform_generation = True

    generator._build_gradle = build_gradle
    generator._main_java = main_java
    generator._gametest_java = gametest_java
    generator._write_item = write_item
    generator._write_block = write_block
    generator.generate = generate


def _migrate_1211_data_tree(project_root: Path, mod_id: str) -> None:
    data = project_root / "src" / "main" / "resources" / "data" / mod_id
    moves = {
        "recipes": "recipe",
        "loot_tables": "loot_table",
        "advancements": "advancement",
    }
    for old_name, new_name in moves.items():
        _merge_directory(data / old_name, data / new_name)
    tags = data / "tags"
    for old_name, new_name in {
        "blocks": "block",
        "items": "item",
        "entity_types": "entity_type",
    }.items():
        _merge_directory(tags / old_name, tags / new_name)

    recipe_dir = data / "recipe"
    if recipe_dir.is_dir():
        for path in recipe_dir.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            result = payload.get("result")
            if isinstance(result, dict) and "item" in result and "id" not in result:
                result["id"] = result.pop("item")
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )


def _merge_directory(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        path.replace(destination)
    shutil.rmtree(source, ignore_errors=True)


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
