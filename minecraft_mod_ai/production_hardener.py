from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    test_files, entrypoints = _gametest_files(
        package_name=info.package_name,
        mod_id=info.mod_id,
        definitions=definitions,
        shard_size=policy.java_shard_size,
    )
    if test_files:
        receipts.append(write_text_files(info, test_files, replace_existing=True))
        entry_receipt = _ensure_gametest_entrypoints(info, entrypoints)
        if entry_receipt.get("status") != "UNCHANGED":
            receipts.append(entry_receipt)

    return {
        "schema_version": "mmm/production-hardening-v1",
        "status": "HARDENED" if receipts else "UNCHANGED",
        "registry_definition_count": len(definitions),
        "gametest_shard_count": len(entrypoints),
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
    extended = root / ".minecraft_ai/extended-modules.json"
    if extended.is_file() and not extended.is_symlink():
        raw = json.loads(extended.read_text(encoding="utf-8"))
        for item in raw.get("modules", []):
            if not isinstance(item, dict):
                continue
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

    gecko = root / ".minecraft_ai/geckolib-entities.json"
    if gecko.is_file() and not gecko.is_symlink():
        raw = json.loads(gecko.read_text(encoding="utf-8"))
        for item in raw.get("entities", []):
            if isinstance(item, dict) and item.get("entity_id"):
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
) -> tuple[dict[str, str], list[str]]:
    files: dict[str, str] = {}
    entries: list[str] = []
    package_path = package_name.replace(".", "/")
    for offset in range(0, len(definitions), shard_size):
        shard = definitions[offset : offset + shard_size]
        index = offset // shard_size
        class_name = f"GeneratedRegistryGameTest{index:04d}"
        entrypoint = f"{package_name}.gametest.{class_name}"
        entries.append(entrypoint)
        relative = f"src/main/java/{package_path}/gametest/{class_name}.java"
        checks = "\n".join(
            f'        require(Registries.{item["registry"]}.containsId(new Identifier("{mod_id}", "{item["id"]}")), "{item["registry"]}:{item["id"]} missing");'
            for item in shard
        )
        files[relative] = f'''package {package_name}.gametest;

import net.fabricmc.fabric.api.gametest.v1.FabricGameTest;
import net.minecraft.registry.Registries;
import net.minecraft.test.GameTest;
import net.minecraft.test.TestContext;
import net.minecraft.util.Identifier;

public final class {class_name} {{
    @GameTest(templateName = FabricGameTest.EMPTY_STRUCTURE)
    public void generatedRegistriesAreLive(TestContext context) {{
{checks}
        context.complete();
    }}

    private static void require(boolean condition, String message) {{
        if (!condition) throw new AssertionError(message);
    }}
}}
'''
    return files, entries


def _ensure_gametest_entrypoints(info, entries: list[str]) -> dict[str, Any]:
    if not entries:
        return {"status": "UNCHANGED", "entries": []}
    metadata = json.loads(info.fabric_mod_json.read_text(encoding="utf-8"))
    entrypoints = metadata.setdefault("entrypoints", {})
    if not isinstance(entrypoints, dict):
        raise ProductionHardeningError("fabric.mod.json entrypoints must be an object.")
    gametest = entrypoints.setdefault("fabric-gametest", [])
    if not isinstance(gametest, list):
        raise ProductionHardeningError("fabric-gametest entrypoints must be a list.")
    existing = {
        item if isinstance(item, str) else item.get("value")
        for item in gametest
        if isinstance(item, (str, dict))
    }
    added = [entry for entry in entries if entry not in existing]
    if not added:
        return {"status": "UNCHANGED", "entries": entries}
    gametest.extend(added)
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
