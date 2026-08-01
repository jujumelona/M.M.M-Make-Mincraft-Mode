from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .extended_content_generator import iter_extended_module_records
from .gametest_discovery import discovered_gametest_root_java
from .geckolib_generator import iter_geckolib_entity_records
from .project_edit import inspect_fabric_project, write_text_files
from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher, sha256_file


class ProductionHardeningError(RuntimeError):
    pass


def harden_generated_project(
    project_root: str | Path,
    *,
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    """Apply deterministic post-generation invariants before validation/build.

    This stage is intentionally model-free. It makes machine blocks render as models,
    emits shard-sized GameTests for every generated registry entry, and registers those
    tests in ``fabric.mod.json``. It is idempotent and operates on manifests rather than
    assuming a small project.
    """

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    info = inspect_fabric_project(project_root)
    receipts: list[dict[str, Any]] = []

    machine_receipt = _ensure_machine_model_render(info)
    if machine_receipt.get("status") != "UNCHANGED":
        receipts.append(machine_receipt)

    definitions = _registry_definitions(info.root)
    test_files, entrypoints, shard_count = _gametest_files(
        package_name=info.package_name,
        mod_id=info.mod_id,
        definitions=definitions,
        shard_size=policy.java_shard_size,
    )
    if test_files:
        write_receipt = write_text_files(
            info,
            test_files,
            replace_existing=True,
        )
        if write_receipt.get("status") != "UNCHANGED":
            receipts.append(write_receipt)
    prune_receipt = _prune_obsolete_gametest_files(
        info,
        active_paths=set(test_files),
    )
    if prune_receipt.get("status") != "UNCHANGED":
        receipts.append(prune_receipt)
    entry_receipt = _ensure_gametest_entrypoints(info, entrypoints)
    if entry_receipt.get("status") != "UNCHANGED":
        receipts.append(entry_receipt)

    return {
        "schema_version": "mmm/production-hardening-v1",
        "status": "HARDENED" if receipts else "UNCHANGED",
        "registry_definition_count": len(definitions),
        "gametest_shard_count": shard_count,
        "receipts": receipts,
    }


def _ensure_machine_model_render(info) -> dict[str, Any]:
    path = (
        info.root
        / "src/main/java"
        / Path(*info.package_name.split("."))
        / "extended/GeneratedExtendedContent.java"
    )
    if not path.is_file() or path.is_symlink():
        return {"status": "UNCHANGED", "reason": "no generated machine root"}
    text = path.read_text(encoding="utf-8")
    if "class GeneratedMachineBlock" not in text:
        return {"status": "UNCHANGED", "reason": "no generated machine block"}
    changed = text
    if "import net.minecraft.block.BlockRenderType;" not in changed:
        anchor = "import net.minecraft.block.BlockState;"
        if anchor not in changed:
            raise ProductionHardeningError("Machine source lacks BlockState import anchor.")
        changed = changed.replace(
            anchor,
            anchor + "\nimport net.minecraft.block.BlockRenderType;",
            1,
        )
    if "public BlockRenderType getRenderType(BlockState state)" not in changed:
        anchor = "        public MachineDefinition definition() { return definition; }"
        if anchor not in changed:
            raise ProductionHardeningError("Machine source lacks definition method anchor.")
        changed = changed.replace(
            anchor,
            anchor
            + "\n\n"
            + "        @Override\n"
            + "        public BlockRenderType getRenderType(BlockState state) {\n"
            + "            return BlockRenderType.MODEL;\n"
            + "        }",
            1,
        )
    if changed == text:
        return {"status": "UNCHANGED", "path": str(path)}
    return TransactionalSourcePatcher(info.root).apply(
        [
            {
                "operation": "replace",
                "path": path.relative_to(info.root).as_posix(),
                "expected_sha256": sha256_file(path),
                "content": changed,
            }
        ]
    )


def _registry_definitions(root: Path) -> list[dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for item in iter_extended_module_records(root):
        module_id = str(item.get("module_id", ""))
        kind = str(item.get("kind", ""))
        registry = {
            "item": "ITEM",
            "food": "ITEM",
            "weapon": "ITEM",
            "tool": "ITEM",
            "armor": "ITEM",
            "block": "BLOCK",
            "machine": "BLOCK",
            "crop": "BLOCK",
            "effect": "STATUS_EFFECT",
            "enchantment": "ENCHANTMENT",
        }.get(kind)
        if registry:
            result[(registry, module_id)] = {
                "registry": registry,
                "id": module_id,
            }
        if kind == "crop":
            result[("ITEM", module_id + "_seeds")] = {
                "registry": "ITEM",
                "id": module_id + "_seeds",
            }

    for item in iter_geckolib_entity_records(root):
        entity_id = str(item["entity_id"])
        result[("ENTITY_TYPE", entity_id)] = {
            "registry": "ENTITY_TYPE",
            "id": entity_id,
        }
    return [result[key] for key in sorted(result)]


def _gametest_files(
    *,
    package_name: str,
    mod_id: str,
    definitions: list[dict[str, str]],
    shard_size: int,
) -> tuple[dict[str, str], list[str], int]:
    if not definitions:
        return {}, [], 0

    files: dict[str, str] = {}
    package_path = package_name.replace(".", "/")
    test_package = f"{package_name}.gametest"
    test_package_path = f"{package_path}/gametest"
    root_class_name = "GeneratedRegistryGameTest"
    unit_prefix = root_class_name + "Unit"
    root_relative = (
        f"src/main/java/{test_package_path}/{root_class_name}.java"
    )
    files[root_relative] = discovered_gametest_root_java(
        package_name=test_package,
        mod_id=mod_id,
        root_class_name=root_class_name,
        unit_class_prefix=unit_prefix,
    )

    shard_count = 0
    for offset in range(0, len(definitions), shard_size):
        shard = definitions[offset : offset + shard_size]
        index = offset // shard_size
        shard_count += 1
        class_name = f"{unit_prefix}{index:04d}"
        relative = f"src/main/java/{test_package_path}/{class_name}.java"
        checks = "\n".join(
            f'        require(Registries.{item["registry"]}.containsId(new Identifier("{mod_id}", "{item["id"]}")), "{item["registry"]}:{item["id"]} missing");'
            for item in shard
        )
        files[relative] = f'''package {test_package};

import net.minecraft.registry.Registries;
import net.minecraft.test.TestContext;
import net.minecraft.util.Identifier;

public final class {class_name} {{
    public static void run(TestContext context) {{
{checks}
    }}

    private static void require(boolean condition, String message) {{
        if (!condition) throw new AssertionError(message);
    }}
}}
'''
    return (
        files,
        [f"{test_package}.{root_class_name}"],
        shard_count,
    )


def _prune_obsolete_gametest_files(
    info,
    *,
    active_paths: set[str],
) -> dict[str, Any]:
    package_directory = (
        info.root
        / "src/main/java"
        / Path(*info.package_name.split("."))
        / "gametest"
    )
    if not package_directory.is_dir():
        return {"status": "UNCHANGED", "removed": []}
    active = {
        (info.root / relative).resolve()
        for relative in active_paths
    }
    obsolete = [
        path
        for path in sorted(package_directory.glob("GeneratedRegistryGameTest*.java"))
        if path.resolve() not in active and path.is_file() and not path.is_symlink()
    ]
    if not obsolete:
        return {"status": "UNCHANGED", "removed": []}
    return TransactionalSourcePatcher(info.root).apply(
        [
            {
                "operation": "delete",
                "path": path.relative_to(info.root).as_posix(),
                "expected_sha256": sha256_file(path),
            }
            for path in obsolete
        ]
    )


def _ensure_gametest_entrypoints(info, entries: list[str]) -> dict[str, Any]:
    metadata = json.loads(info.fabric_mod_json.read_text(encoding="utf-8"))
    entrypoints = metadata.setdefault("entrypoints", {})
    if not isinstance(entrypoints, dict):
        raise ProductionHardeningError("fabric.mod.json entrypoints must be an object.")
    gametest = entrypoints.setdefault("fabric-gametest", [])
    if not isinstance(gametest, list):
        raise ProductionHardeningError("fabric-gametest entrypoints must be a list.")
    generated_prefix = f"{info.package_name}.gametest.GeneratedRegistryGameTest"

    def entrypoint_value(item: Any) -> str | None:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            value = item.get("value")
            return value if isinstance(value, str) else None
        return None

    retained = [
        item
        for item in gametest
        if not (
            (value := entrypoint_value(item))
            and value.startswith(generated_prefix)
        )
    ]
    replacement = retained + entries
    if replacement == gametest:
        return {"status": "UNCHANGED", "entries": entries}
    entrypoints["fabric-gametest"] = replacement
    return TransactionalSourcePatcher(info.root).apply(
        [
            {
                "operation": "replace",
                "path": "src/main/resources/fabric.mod.json",
                "expected_sha256": sha256_file(info.fabric_mod_json),
                "content": json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            }
        ]
    )
