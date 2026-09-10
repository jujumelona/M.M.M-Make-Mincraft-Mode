from __future__ import annotations

"""Cross-artifact reference and coherence validators.

Verifies that Java registrations, ResourceKeys, client item definitions,
item models, textures, and lang translations form a consistent, complete graph.
"""

import json
from pathlib import Path
import re
from typing import Any


class ReferenceValidationError(ValueError):
    pass


def validate_item_vertical_slice(
    workspace_dir: Path | str,
    *,
    mod_id: str,
    item_name: str,
    package_name: str,
    main_class: str = "",
    expected_stack_limit: int | None = None,
) -> dict[str, Any]:
    """Validate that an item vertical slice is complete, connected, and coherent across all files."""
    root = Path(workspace_dir)
    pkg_path = package_name.replace(".", "/")
    constant_name = "".join(c if c.isalnum() else "_" for c in item_name).strip("_").upper()
    main_class_name = main_class or "".join(part.capitalize() for part in mod_id.split("_")) + "Mod"

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    unresolved_obligations: list[str] = []

    # 1. Check ModItemIds.java
    ids_file = root / "src" / "main" / "java" / pkg_path / "registry" / "ModItemIds.java"
    if not ids_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: ModItemIds.java not found at {ids_file}")
    ids_content = ids_file.read_text(encoding="utf-8")
    key_pattern = rf"ResourceKey<Item>\s+{constant_name}_KEY\s*="
    if not re.search(key_pattern, ids_content):
        raise ReferenceValidationError(
            f"REF_KEY_MISSING: ModItemIds.java does not declare ResourceKey<Item> {constant_name}_KEY"
        )
    if f'"{item_name}"' not in ids_content:
        raise ReferenceValidationError(
            f"REF_KEY_PATH_MISMATCH: ModItemIds.java does not reference registry path {item_name!r}"
        )
    checks["resource_key_declared"] = True

    # 2. Check ModItems.java
    items_file = root / "src" / "main" / "java" / pkg_path / "registry" / "ModItems.java"
    if not items_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: ModItems.java not found at {items_file}")
    items_content = items_file.read_text(encoding="utf-8")
    if "BuiltInRegistries.ITEM" not in items_content:
        raise ReferenceValidationError("REF_REGISTRY_MISSING: ModItems.java must use BuiltInRegistries.ITEM")
    if f"ModItemIds.{constant_name}_KEY" not in items_content:
        raise ReferenceValidationError(
            f"REF_REGISTRY_KEY_UNLINKED: ModItems.java does not register using ModItemIds.{constant_name}_KEY"
        )
    if expected_stack_limit is not None:
        expected_stack_clause = f".stacksTo({expected_stack_limit})"
        if expected_stack_clause not in items_content:
            raise ReferenceValidationError(
                f"REF_PROPERTY_MISSING: ModItems.java does not include expected {expected_stack_clause}"
            )
        checks["stack_limit_verified"] = True
    checks["item_registered"] = True

    # 3. Check Main Initializer
    main_file = root / "src" / "main" / "java" / pkg_path / f"{main_class_name}.java"
    if main_file.is_file():
        main_content = main_file.read_text(encoding="utf-8")
        if "ModItems.initialize()" not in main_content:
            raise ReferenceValidationError(
                f"REF_INIT_MISSING: Main class {main_file.name} does not invoke ModItems.initialize()"
            )
        checks["main_initializer_hooked"] = True
    else:
        checks["main_initializer_hooked"] = False

    # 4. Check Client Item JSON (assets/<modid>/items/<item>.json)
    client_item_file = root / "src" / "main" / "resources" / "assets" / mod_id / "items" / f"{item_name}.json"
    if not client_item_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: Client item JSON not found at {client_item_file}")
    try:
        client_data = json.loads(client_item_file.read_text(encoding="utf-8"))
        expected_model_ref = f"{mod_id}:item/{item_name}"
        actual_model_ref = client_data.get("model", {}).get("model")
        if actual_model_ref != expected_model_ref:
            raise ReferenceValidationError(
                f"REF_CLIENT_ITEM_MODEL_MISMATCH: Client item references {actual_model_ref!r}, expected {expected_model_ref!r}"
            )
        checks["client_item_model_matched"] = True
    except json.JSONDecodeError as exc:
        raise ReferenceValidationError(f"REF_INVALID_JSON: Client item JSON is invalid: {exc}") from exc

    # 5. Check Item Model JSON (assets/<modid>/models/item/<item>.json)
    model_file = root / "src" / "main" / "resources" / "assets" / mod_id / "models" / "item" / f"{item_name}.json"
    if not model_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: Item model JSON not found at {model_file}")
    try:
        model_data = json.loads(model_file.read_text(encoding="utf-8"))
        layer0 = model_data.get("textures", {}).get("layer0")
        expected_texture = f"{mod_id}:item/{item_name}"
        if layer0 != expected_texture:
            raise ReferenceValidationError(
                f"REF_MODEL_TEXTURE_MISMATCH: Model layer0 is {layer0!r}, expected {expected_texture!r}"
            )
        checks["item_model_matched"] = True
    except json.JSONDecodeError as exc:
        raise ReferenceValidationError(f"REF_INVALID_JSON: Item model JSON is invalid: {exc}") from exc

    # 6. Check Lang File (assets/<modid>/lang/en_us.json)
    lang_file = root / "src" / "main" / "resources" / "assets" / mod_id / "lang" / "en_us.json"
    if not lang_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: Lang file not found at {lang_file}")
    try:
        lang_data = json.loads(lang_file.read_text(encoding="utf-8"))
        lang_key = f"item.{mod_id}.{item_name}"
        if lang_key not in lang_data:
            raise ReferenceValidationError(f"REF_LANG_MISSING: Lang entry {lang_key!r} not found in {lang_file.name}")
        checks["lang_entry_present"] = True
    except json.JSONDecodeError as exc:
        raise ReferenceValidationError(f"REF_INVALID_JSON: Lang JSON is invalid: {exc}") from exc

    # 7. Check Texture asset or record obligation
    texture_file = root / "src" / "main" / "resources" / "assets" / mod_id / "textures" / "item" / f"{item_name}.png"
    if texture_file.is_file():
        checks["texture_asset_present"] = True
    else:
        checks["texture_asset_present"] = False
        unresolved_obligations.append(f"{mod_id}:textures/item/{item_name}.png")

    details["checks"] = checks
    details["unresolved_asset_obligations"] = unresolved_obligations

    return {
        "status": "PASS",
        "item": item_name,
        "mod_id": mod_id,
        "details": details,
    }


def validate_block_vertical_slice(
    workspace_dir: Path | str,
    *,
    mod_id: str,
    block_name: str,
    package_name: str,
    main_class: str = "",
    expected_drop_item: str | None = None,
) -> dict[str, Any]:
    """Validate that a block vertical slice is complete, connected, and coherent across all files."""
    root = Path(workspace_dir)
    pkg_path = package_name.replace(".", "/")
    constant_name = "".join(c if c.isalnum() else "_" for c in block_name).strip("_").upper()
    main_class_name = main_class or "".join(part.capitalize() for part in mod_id.split("_")) + "Mod"

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    unresolved_obligations: list[str] = []

    # 1. Check ModBlockIds.java
    ids_file = root / "src" / "main" / "java" / pkg_path / "registry" / "ModBlockIds.java"
    if not ids_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: ModBlockIds.java not found at {ids_file}")
    ids_content = ids_file.read_text(encoding="utf-8")
    key_pattern = rf"ResourceKey<Block>\s+{constant_name}_KEY\s*="
    if not re.search(key_pattern, ids_content):
        raise ReferenceValidationError(
            f"REF_KEY_MISSING: ModBlockIds.java does not declare ResourceKey<Block> {constant_name}_KEY"
        )
    if f'"{block_name}"' not in ids_content:
        raise ReferenceValidationError(
            f"REF_KEY_PATH_MISMATCH: ModBlockIds.java does not reference registry path {block_name!r}"
        )
    checks["resource_key_declared"] = True

    # 2. Check ModBlocks.java
    blocks_file = root / "src" / "main" / "java" / pkg_path / "registry" / "ModBlocks.java"
    if not blocks_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: ModBlocks.java not found at {blocks_file}")
    blocks_content = blocks_file.read_text(encoding="utf-8")
    if "BuiltInRegistries.BLOCK" not in blocks_content:
        raise ReferenceValidationError("REF_REGISTRY_MISSING: ModBlocks.java must use BuiltInRegistries.BLOCK")
    if f"ModBlockIds.{constant_name}_KEY" not in blocks_content:
        raise ReferenceValidationError(
            f"REF_REGISTRY_KEY_UNLINKED: ModBlocks.java does not register using ModBlockIds.{constant_name}_KEY"
        )
    checks["block_registered"] = True

    # 3. Check Main Initializer
    main_file = root / "src" / "main" / "java" / pkg_path / f"{main_class_name}.java"
    if main_file.is_file():
        main_content = main_file.read_text(encoding="utf-8")
        if "ModBlocks.initialize()" not in main_content:
            raise ReferenceValidationError(
                f"REF_INIT_MISSING: Main class {main_file.name} does not invoke ModBlocks.initialize()"
            )
        checks["main_initializer_hooked"] = True
    else:
        checks["main_initializer_hooked"] = False

    # 4. Check Blockstate JSON (assets/<modid>/blockstates/<block>.json)
    blockstate_file = root / "src" / "main" / "resources" / "assets" / mod_id / "blockstates" / f"{block_name}.json"
    if not blockstate_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: Blockstate JSON not found at {blockstate_file}")
    try:
        bs_data = json.loads(blockstate_file.read_text(encoding="utf-8"))
        expected_model_ref = f"{mod_id}:block/{block_name}"
        actual_model_ref = bs_data.get("variants", {}).get("", {}).get("model")
        if actual_model_ref != expected_model_ref:
            raise ReferenceValidationError(
                f"REF_BLOCKSTATE_MODEL_MISMATCH: Blockstate references {actual_model_ref!r}, expected {expected_model_ref!r}"
            )
        checks["blockstate_model_matched"] = True
    except json.JSONDecodeError as exc:
        raise ReferenceValidationError(f"REF_INVALID_JSON: Blockstate JSON is invalid: {exc}") from exc

    # 5. Check Block Model JSON (assets/<modid>/models/block/<block>.json)
    model_file = root / "src" / "main" / "resources" / "assets" / mod_id / "models" / "block" / f"{block_name}.json"
    if not model_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: Block model JSON not found at {model_file}")
    try:
        model_data = json.loads(model_file.read_text(encoding="utf-8"))
        all_texture = model_data.get("textures", {}).get("all")
        expected_texture = f"{mod_id}:block/{block_name}"
        if all_texture != expected_texture:
            raise ReferenceValidationError(
                f"REF_MODEL_TEXTURE_MISMATCH: Block model all texture is {all_texture!r}, expected {expected_texture!r}"
            )
        checks["block_model_matched"] = True
    except json.JSONDecodeError as exc:
        raise ReferenceValidationError(f"REF_INVALID_JSON: Block model JSON is invalid: {exc}") from exc

    # 6. Check Lang File (assets/<modid>/lang/en_us.json)
    lang_file = root / "src" / "main" / "resources" / "assets" / mod_id / "lang" / "en_us.json"
    if not lang_file.is_file():
        raise ReferenceValidationError(f"REF_MISSING_FILE: Lang file not found at {lang_file}")
    try:
        lang_data = json.loads(lang_file.read_text(encoding="utf-8"))
        lang_key = f"block.{mod_id}.{block_name}"
        if lang_key not in lang_data:
            raise ReferenceValidationError(f"REF_LANG_MISSING: Lang entry {lang_key!r} not found in {lang_file.name}")
        checks["lang_entry_present"] = True
    except json.JSONDecodeError as exc:
        raise ReferenceValidationError(f"REF_INVALID_JSON: Lang JSON is invalid: {exc}") from exc

    # 7. Check Texture asset or record obligation
    texture_file = root / "src" / "main" / "resources" / "assets" / mod_id / "textures" / "block" / f"{block_name}.png"
    if texture_file.is_file():
        checks["texture_asset_present"] = True
    else:
        checks["texture_asset_present"] = False
        unresolved_obligations.append(f"{mod_id}:textures/block/{block_name}.png")

    # 8. Check Loot Table if drop item specified
    if expected_drop_item:
        loot_file = root / "src" / "main" / "resources" / "data" / mod_id / "loot_tables" / "blocks" / f"{block_name}.json"
        if not loot_file.is_file():
            raise ReferenceValidationError(f"REF_MISSING_FILE: Loot table JSON not found at {loot_file}")
        try:
            loot_data = json.loads(loot_file.read_text(encoding="utf-8"))
            pools = loot_data.get("pools", [])
            found_drop = False
            expected_entry_name = f"{mod_id}:{expected_drop_item}"
            for pool in pools:
                for entry in pool.get("entries", []):
                    if entry.get("name") == expected_entry_name:
                        found_drop = True
                        break
            if not found_drop:
                raise ReferenceValidationError(
                    f"REF_LOOT_DROP_MISMATCH: Loot table {loot_file.name} does not drop {expected_entry_name!r}"
                )
            checks["loot_drop_matched"] = True
        except json.JSONDecodeError as exc:
            raise ReferenceValidationError(f"REF_INVALID_JSON: Loot table JSON is invalid: {exc}") from exc

    details["checks"] = checks
    details["unresolved_asset_obligations"] = unresolved_obligations

    return {
        "status": "PASS",
        "block": block_name,
        "mod_id": mod_id,
        "details": details,
    }
