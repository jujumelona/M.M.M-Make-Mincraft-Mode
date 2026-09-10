from __future__ import annotations

"""Deterministic file materializer with before/after SHA-256 receipts.

Supports four materialization modes:
1. whole_file: Writes complete source or resource file.
2. java_patch: Inserts code at exact anchor coordinates (e.g. /* MMM:properties:raw_lunite */).
3. json_merge: Merges JSON fragments (e.g. lang file entries).
4. binary_asset: Writes raw binary data (e.g. textures).
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifact_job import ArtifactJob


class MaterializeError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializeReceipt:
    target_path: str
    operation: str
    before_sha256: str | None
    after_sha256: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return self.target_path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_path(path: Path | str, base_dir: Path | None = None) -> Path:
    p = Path(path)
    if base_dir is not None and not p.is_absolute():
        p = base_dir / p
    if base_dir is not None and not p.resolve().is_relative_to(Path(base_dir).resolve()):
        raise MaterializeError(f"TARGET_ESCAPE: {p}")
    return p


def materialize_whole_file(
    target_path: Path | str,
    content: str | bytes,
    *,
    base_dir: Path | None = None,
    expected_sha256: str | None = None,
) -> MaterializeReceipt:
    """Write complete file to target path and return SHA256 receipt."""
    p = _resolve_path(target_path, base_dir)
    data = content.encode("utf-8") if isinstance(content, str) else content

    before_sha: str | None = None
    if p.is_file():
        before_sha = _sha256(p.read_bytes())
        if expected_sha256 is not None and before_sha != expected_sha256:
            raise MaterializeError(
                f"SHA_MISMATCH: Target file {p} sha256 is {before_sha}, expected {expected_sha256}"
            )

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    after_sha = _sha256(data)

    return MaterializeReceipt(
        target_path=str(p),
        operation="whole_file",
        before_sha256=before_sha,
        after_sha256=after_sha,
        status="SUCCESS",
        details={"bytes_written": len(data)},
    )


def materialize_java_patch(
    target_path: Path | str,
    anchor: str,
    patch: str,
    *,
    base_dir: Path | None = None,
    keep_anchor: bool = True,
) -> MaterializeReceipt:
    """Patch Java source at exact anchor coordinate."""
    p = _resolve_path(target_path, base_dir)
    if not p.is_file():
        raise MaterializeError(f"PATCH_TARGET_NOT_FOUND: Target file {p} does not exist for anchor {anchor!r}")

    original_bytes = p.read_bytes()
    before_sha = _sha256(original_bytes)
    original_text = original_bytes.decode("utf-8")

    anchor_count = original_text.count(anchor)
    if anchor_count == 0:
        raise MaterializeError(f"ANCHOR_NOT_FOUND: Anchor {anchor!r} not found in {p}")
    if anchor_count > 1:
        raise MaterializeError(
            f"ANCHOR_AMBIGUOUS: Anchor {anchor!r} found {anchor_count} times in {p}; ambiguous patch location"
        )

    if keep_anchor:
        # Check indentation of the anchor line
        replacement = f"{patch.strip()}\n    {anchor}"
    else:
        replacement = patch.strip()

    updated_text = original_text.replace(anchor, replacement, 1)
    updated_bytes = updated_text.encode("utf-8")
    p.write_bytes(updated_bytes)
    after_sha = _sha256(updated_bytes)

    return MaterializeReceipt(
        target_path=str(p),
        operation="java_patch",
        before_sha256=before_sha,
        after_sha256=after_sha,
        status="SUCCESS",
        details={"anchor": anchor, "keep_anchor": keep_anchor},
    )


def _deep_merge_dicts(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in updates.items():
        if k in merged:
            if isinstance(merged[k], dict) and isinstance(v, Mapping):
                merged[k] = _deep_merge_dicts(merged[k], v)
            elif merged[k] != v:
                raise MaterializeError(
                    f"JSON_KEY_CONFLICT: Conflicting value for key {k!r}: existing {merged[k]!r} vs new {v!r}"
                )
            else:
                merged[k] = v
        else:
            merged[k] = v
    return merged


def materialize_json_merge(
    target_path: Path | str,
    fragment: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> MaterializeReceipt:
    """Merge JSON fragment into existing JSON resource or create a new one."""
    p = _resolve_path(target_path, base_dir)
    before_sha: str | None = None
    existing_data: dict[str, Any] = {}

    if p.is_file():
        raw_bytes = p.read_bytes()
        before_sha = _sha256(raw_bytes)
        if raw_bytes.strip():
            try:
                parsed = json.loads(raw_bytes.decode("utf-8"))
            except Exception as exc:
                raise MaterializeError(f"JSON_CORRUPTED: Failed to parse existing JSON {p}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise MaterializeError(f"JSON_CORRUPTED: Existing JSON {p} root is not an object: {type(parsed)}")
            existing_data = parsed

    merged = _deep_merge_dicts(existing_data, fragment)
    formatted = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
    updated_bytes = formatted.encode("utf-8")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(updated_bytes)
    after_sha = _sha256(updated_bytes)

    return MaterializeReceipt(
        target_path=str(p),
        operation="json_merge",
        before_sha256=before_sha,
        after_sha256=after_sha,
        status="SUCCESS",
        details={"keys_count": len(merged)},
    )


def materialize_binary_asset(
    target_path: Path | str,
    data: bytes,
    *,
    base_dir: Path | None = None,
) -> MaterializeReceipt:
    """Write binary asset (such as PNG texture)."""
    p = _resolve_path(target_path, base_dir)
    before_sha: str | None = None
    if p.is_file():
        before_sha = _sha256(p.read_bytes())

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    after_sha = _sha256(data)

    return MaterializeReceipt(
        target_path=str(p),
        operation="binary_asset",
        before_sha256=before_sha,
        after_sha256=after_sha,
        status="SUCCESS",
        details={"bytes_written": len(data)},
    )


def materialize_job_output(
    job: ArtifactJob,
    rendered_content: Any,
    *,
    base_dir: Path | None = None,
) -> MaterializeReceipt:
    """Materialize rendered ArtifactJob output according to its target and anchor."""
    target_file = job.target_path
    if not target_file:
        raise MaterializeError(f"JOB_NO_TARGET: Job {job.job_id} has no target_path")

    operation = job.operation
    if operation and operation not in {"CREATE_FILE", "REPLACE_FILE", "JAVA_PATCH", "JSON_OBJECT_MERGE", "JSON_ARRAY_MERGE", "BINARY_WRITE"}:
        raise MaterializeError(f"JOB_OPERATION_UNSUPPORTED: {operation}")
    p = _resolve_path(target_file, base_dir)
    if base_dir is not None:
        root = Path(base_dir).resolve()
        for origin, other in (("src/main/resources", "src/main/generated"), ("src/main/generated", "src/main/resources")):
            source = root / origin
            if p.resolve().is_relative_to(source):
                counterpart = root / other / p.resolve().relative_to(source)
                if counterpart.is_file():
                    raise MaterializeError(f"RESOURCE_OWNERSHIP_CONFLICT: {p} and {counterpart}")
    if operation and operation != "JAVA_PATCH" and job.anchor:
        raise MaterializeError("OPERATION_ANCHOR_CONFLICT")
    if operation == "BINARY_WRITE" and not isinstance(rendered_content, bytes):
        raise MaterializeError("BINARY_BYTES_REQUIRED")
    if operation == "REPLACE_FILE":
        if not p.is_file() or not job.expected_sha256:
            raise MaterializeError("REPLACE_REQUIRES_BEFORE_HASH")
        return materialize_whole_file(target_file, rendered_content, base_dir=base_dir, expected_sha256=job.expected_sha256)
    if operation == "JSON_ARRAY_MERGE":
        incoming = json.loads(rendered_content) if isinstance(rendered_content, str) else rendered_content
        current = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []
        if not isinstance(incoming, list) or not isinstance(current, list):
            raise MaterializeError("JSON_ARRAY_REQUIRED")
        merged = list(current)
        for value in incoming:
            if value not in merged:
                merged.append(value)
        return materialize_whole_file(p, json.dumps(merged, ensure_ascii=False)+"\n", base_dir=base_dir)
    if operation in {"CREATE_FILE", "BINARY_WRITE"} and p.is_file():
        incoming = rendered_content if isinstance(rendered_content, bytes) else (
            json.dumps(rendered_content, indent=2, ensure_ascii=False)+"\n"
            if isinstance(rendered_content, (dict,list)) else str(rendered_content)
        ).encode("utf-8")
        if p.read_bytes() != incoming:
            raise MaterializeError(f"EXCLUSIVE_FILE_CONFLICT: {p}")
    # If anchor is specified, perform java_patch
    if operation == "JAVA_PATCH" and not job.anchor:
        raise MaterializeError("JAVA_ANCHOR_REQUIRED")
    if job.anchor:
        # Settings properties are per-item and should replace the anchor
        is_property_anchor = "properties:" in job.anchor
        patch_str = str(rendered_content)
        return materialize_java_patch(
            target_file,
            job.anchor,
            patch_str,
            base_dir=base_dir,
            keep_anchor=not is_property_anchor,
        )

    # If it's a JSON fragment (e.g. lang file), perform json_merge
    if operation == "JSON_OBJECT_MERGE" or (not operation and target_file.endswith(".json") and (isinstance(rendered_content, Mapping) or "lang" in job.template_id)):
        if isinstance(rendered_content, str):
            fragment = json.loads(rendered_content)
        else:
            fragment = dict(rendered_content)
        return materialize_json_merge(target_file, fragment, base_dir=base_dir)

    # Otherwise whole_file
    if isinstance(rendered_content, bytes):
        return materialize_binary_asset(target_file, rendered_content, base_dir=base_dir)

    content_str = (
        json.dumps(rendered_content, indent=2, ensure_ascii=False) + "\n"
        if isinstance(rendered_content, (dict, list))
        else str(rendered_content)
    )
    return materialize_whole_file(target_file, content_str, base_dir=base_dir)


def ensure_artifact_scaffolding(
    project_root: Path | str,
    *,
    mod_id: str,
    package_name: str,
    main_class: str = "",
) -> None:
    """Ensure prerequisite skeleton Java/resource files and anchors exist for leaf template execution."""
    root = Path(project_root).resolve()
    pkg_path = package_name.replace(".", "/")
    main_class_name = (
        main_class or "".join(part.capitalize() for part in mod_id.split("_")) + "Mod"
    )

    # 1. ModItemIds.java
    ids_path = root / "src" / "main" / "java" / pkg_path / "registry" / "ModItemIds.java"
    if not ids_path.is_file():
        mod_item_ids_skeleton = (
            f"package {package_name}.registry;\n\n"
            "import net.minecraft.core.registries.Registries;\n"
            "import net.minecraft.resources.ResourceKey;\n"
            "import net.minecraft.resources.Identifier;\n"
            "import net.minecraft.world.item.Item;\n\n"
            "public final class ModItemIds {\n"
            "    private ModItemIds() {}\n\n"
            "    /* MMM:item_keys */\n"
            "}\n"
        )
        materialize_whole_file(ids_path, mod_item_ids_skeleton)
    else:
        content = ids_path.read_text(encoding="utf-8")
        if "/* MMM:item_keys */" not in content:
            last_brace = content.rfind("}")
            if last_brace != -1:
                content = content[:last_brace] + "    /* MMM:item_keys */\n" + content[last_brace:]
                ids_path.write_text(content, encoding="utf-8")

    # 2. ModItems.java
    items_path = root / "src" / "main" / "java" / pkg_path / "registry" / "ModItems.java"
    if not items_path.is_file():
        mod_items_skeleton = (
            f"package {package_name}.registry;\n\n"
            "import net.minecraft.core.Registry;\n"
            "import net.minecraft.core.registries.BuiltInRegistries;\n"
            "import net.minecraft.world.item.Item;\n\n"
            "public final class ModItems {\n"
            "    private ModItems() {}\n\n"
            "    /* MMM:item_registry */\n\n"
            "    public static void initialize() {}\n"
            "}\n"
        )
        materialize_whole_file(items_path, mod_items_skeleton)
    else:
        content = items_path.read_text(encoding="utf-8")
        if "/* MMM:item_registry */" not in content:
            last_brace = content.rfind("}")
            if last_brace != -1:
                content = content[:last_brace] + "    /* MMM:item_registry */\n" + content[last_brace:]
                items_path.write_text(content, encoding="utf-8")

    # 3. ModBlockIds.java
    block_ids_path = root / "src" / "main" / "java" / pkg_path / "registry" / "ModBlockIds.java"
    if not block_ids_path.is_file():
        mod_block_ids_skeleton = (
            f"package {package_name}.registry;\n\n"
            "import net.minecraft.core.registries.Registries;\n"
            "import net.minecraft.resources.ResourceKey;\n"
            "import net.minecraft.resources.Identifier;\n"
            "import net.minecraft.world.level.block.Block;\n\n"
            "public final class ModBlockIds {\n"
            "    private ModBlockIds() {}\n\n"
            "    /* MMM:block_keys */\n"
            "}\n"
        )
        materialize_whole_file(block_ids_path, mod_block_ids_skeleton)
    else:
        content = block_ids_path.read_text(encoding="utf-8")
        if "/* MMM:block_keys */" not in content:
            last_brace = content.rfind("}")
            if last_brace != -1:
                content = content[:last_brace] + "    /* MMM:block_keys */\n" + content[last_brace:]
                block_ids_path.write_text(content, encoding="utf-8")

    # 4. ModBlocks.java
    blocks_path = root / "src" / "main" / "java" / pkg_path / "registry" / "ModBlocks.java"
    if not blocks_path.is_file():
        mod_blocks_skeleton = (
            f"package {package_name}.registry;\n\n"
            "import net.minecraft.core.Registry;\n"
            "import net.minecraft.core.registries.BuiltInRegistries;\n"
            "import net.minecraft.world.level.block.Block;\n\n"
            "public final class ModBlocks {\n"
            "    private ModBlocks() {}\n\n"
            "    /* MMM:block_registry */\n\n"
            "    public static void initialize() {}\n"
            "}\n"
        )
        materialize_whole_file(blocks_path, mod_blocks_skeleton)
    else:
        content = blocks_path.read_text(encoding="utf-8")
        if "/* MMM:block_registry */" not in content:
            last_brace = content.rfind("}")
            if last_brace != -1:
                content = content[:last_brace] + "    /* MMM:block_registry */\n" + content[last_brace:]
                blocks_path.write_text(content, encoding="utf-8")

    # 5. Main class initializer
    main_path = root / "src" / "main" / "java" / pkg_path / f"{main_class_name}.java"
    if not main_path.is_file():
        main_skeleton = (
            f"package {package_name};\n\n"
            "import net.fabricmc.api.ModInitializer;\n"
            f"import {package_name}.registry.ModBlocks;\n"
            f"import {package_name}.registry.ModItems;\n\n"
            f"public final class {main_class_name} implements ModInitializer {{\n"
            f'    public static final String MOD_ID = "{mod_id}";\n\n'
            "    @Override\n"
            "    public void onInitialize() {\n"
            "        ModBlocks.initialize();\n"
            "        ModItems.initialize();\n"
            "        /* MMM:init */\n"
            "    }\n"
            "}\n"
        )
        materialize_whole_file(main_path, main_skeleton)
    else:
        main_text = main_path.read_text(encoding="utf-8")
        dirty = False
        for reg_name in ("ModBlocks", "ModItems"):
            import_stmt = f"import {package_name}.registry.{reg_name};"
            if import_stmt not in main_text:
                pkg_decl = f"package {package_name};"
                if pkg_decl in main_text:
                    main_text = main_text.replace(
                        pkg_decl,
                        f"{pkg_decl}\n\n{import_stmt}",
                        1,
                    )
                    dirty = True
        for init_call in ("ModBlocks.initialize();", "ModItems.initialize();"):
            if init_call not in main_text:
                if "/* MMM:init */" in main_text:
                    main_text = main_text.replace(
                        "/* MMM:init */",
                        f"{init_call}\n        /* MMM:init */",
                        1,
                    )
                    dirty = True
        if "/* MMM:init */" not in main_text:
            init_idx = main_text.find("onInitialize()")
            if init_idx != -1:
                brace_idx = main_text.find("{", init_idx)
                if brace_idx != -1:
                    main_text = (
                        main_text[: brace_idx + 1]
                        + "\n        ModBlocks.initialize();\n        ModItems.initialize();\n        /* MMM:init */"
                        + main_text[brace_idx + 1 :]
                    )
                    dirty = True
        if dirty:
            main_path.write_text(main_text, encoding="utf-8")

    # 6. en_us.json
    lang_path = root / "src" / "main" / "resources" / "assets" / mod_id / "lang" / "en_us.json"
    if not lang_path.is_file():
        lang_path.parent.mkdir(parents=True, exist_ok=True)
        lang_path.write_text("{}\n", encoding="utf-8")

