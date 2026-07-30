from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .source_patch import TransactionalSourcePatcher, sha256_file


class ProjectEditError(RuntimeError):
    pass


@dataclass(frozen=True)
class FabricProjectInfo:
    root: Path
    mod_id: str
    main_entrypoint: str
    package_name: str
    main_class: str
    main_java: Path
    fabric_mod_json: Path
    main_entrypoints: tuple[str, ...] = ()


def inspect_fabric_project(project_root: str | Path) -> FabricProjectInfo:
    root = Path(project_root).expanduser().resolve()
    metadata = root / "src/main/resources/fabric.mod.json"
    if not metadata.is_file() or metadata.is_symlink():
        raise ProjectEditError("fabric.mod.json is missing from the generated project.")
    raw = json.loads(metadata.read_text(encoding="utf-8"))
    mod_id = raw.get("id")
    entrypoints = raw.get("entrypoints", {})
    main_raw = entrypoints.get("main", []) if isinstance(entrypoints, dict) else []
    if not isinstance(mod_id, str) or not isinstance(main_raw, list) or not main_raw:
        raise ProjectEditError("Expected at least one Fabric main entrypoint.")

    main_values: list[str] = []
    for entry in main_raw:
        if isinstance(entry, str):
            value = entry
        elif isinstance(entry, dict) and isinstance(entry.get("value"), str):
            value = str(entry["value"])
        else:
            raise ProjectEditError("Fabric main entrypoint is invalid.")
        if "." not in value:
            raise ProjectEditError("Fabric main entrypoint is invalid.")
        main_values.append(value)

    selected = main_values[0]
    selected_java: Path | None = None
    for candidate in main_values:
        package, class_name = candidate.rsplit(".", 1)
        java = root / "src/main/java" / Path(*package.split(".")) / f"{class_name}.java"
        if java.is_file() and not java.is_symlink():
            selected = candidate
            selected_java = java
            break
    package_name, main_class = selected.rsplit(".", 1)
    main_java = selected_java or (
        root
        / "src/main/java"
        / Path(*package_name.split("."))
        / f"{main_class}.java"
    )
    return FabricProjectInfo(
        root=root,
        mod_id=mod_id,
        main_entrypoint=selected,
        package_name=package_name,
        main_class=main_class,
        main_java=main_java,
        fabric_mod_json=metadata,
        main_entrypoints=tuple(main_values),
    )


def ensure_main_initializer_call(
    info: FabricProjectInfo,
    *,
    import_line: str,
    call_line: str,
    marker: str,
) -> dict[str, Any]:
    import_line = import_line.rstrip(";") + ";"
    call_line = call_line.rstrip(";") + ";"
    if info.main_java.is_file() and not info.main_java.is_symlink():
        text = info.main_java.read_text(encoding="utf-8")
        changed = _insert_import(text, import_line)
        changed, inserted = _insert_initializer_call(changed, call_line, marker)
        if inserted:
            if changed == text:
                return {"status": "UNCHANGED", "path": str(info.main_java)}
            relative = info.main_java.relative_to(info.root).as_posix()
            return TransactionalSourcePatcher(info.root).apply(
                [
                    {
                        "operation": "replace",
                        "path": relative,
                        "expected_sha256": sha256_file(info.main_java),
                        "content": changed,
                    }
                ]
            )
    return _ensure_generated_initializer(
        info,
        import_line=import_line,
        call_line=call_line,
        marker=marker,
    )


def _insert_import(text: str, import_line: str) -> str:
    if import_line in text:
        return text
    imports = list(re.finditer(r"(?m)^import\s+[^;]+;\s*$", text))
    if imports:
        position = imports[-1].end()
        return text[:position] + "\n" + import_line + text[position:]
    package = re.search(r"(?m)^package\s+[^;]+;\s*$", text)
    if package:
        position = package.end()
        return text[:position] + "\n\n" + import_line + text[position:]
    raise ProjectEditError("Could not locate Java package/import insertion point.")


def _insert_initializer_call(
    text: str,
    call_line: str,
    marker: str,
) -> tuple[str, bool]:
    marker_line = f"        // MMM:{marker}"
    if marker_line in text:
        return text, True
    method = re.search(
        r"(?m)^(?P<indent>[ \t]*)(?:public\s+)?void\s+onInitialize\s*\(\s*\)\s*\{",
        text,
    )
    if not method:
        return text, False
    indent = method.group("indent") + "    "
    rendered = f"\n{indent}// MMM:{marker}\n{indent}{call_line}"
    position = method.end()
    return text[:position] + rendered + text[position:], True


def _ensure_generated_initializer(
    info: FabricProjectInfo,
    *,
    import_line: str,
    call_line: str,
    marker: str,
) -> dict[str, Any]:
    bridge_package = info.package_name + ".mmm_generated"
    entrypoint = bridge_package + ".MmmGeneratedInitializer"
    relative = (
        "src/main/java/"
        + bridge_package.replace(".", "/")
        + "/MmmGeneratedInitializer.java"
    )
    path = info.root / relative
    if path.is_file() and not path.is_symlink():
        source = path.read_text(encoding="utf-8")
    else:
        source = f'''package {bridge_package};

import net.fabricmc.api.ModInitializer;
// MMM:GENERATED_IMPORTS

public final class MmmGeneratedInitializer implements ModInitializer {{
    @Override
    public void onInitialize() {{
        // MMM:GENERATED_CALLS
    }}
}}
'''
    if import_line not in source:
        source = source.replace(
            "// MMM:GENERATED_IMPORTS",
            import_line + "\n// MMM:GENERATED_IMPORTS",
            1,
        )
    marker_line = f"        // MMM:{marker}"
    if marker_line not in source:
        source = source.replace(
            "        // MMM:GENERATED_CALLS",
            marker_line
            + "\n        "
            + call_line
            + "\n        // MMM:GENERATED_CALLS",
            1,
        )
    operations: list[dict[str, Any]] = []
    if path.is_file():
        current = path.read_text(encoding="utf-8")
        if current != source:
            operations.append(
                {
                    "operation": "replace",
                    "path": relative,
                    "expected_sha256": sha256_file(path),
                    "content": source,
                }
            )
    else:
        operations.append(
            {"operation": "create", "path": relative, "content": source}
        )

    metadata = json.loads(info.fabric_mod_json.read_text(encoding="utf-8"))
    entrypoints = metadata.setdefault("entrypoints", {})
    if not isinstance(entrypoints, dict):
        raise ProjectEditError("fabric.mod.json entrypoints must be an object.")
    main = entrypoints.setdefault("main", [])
    if not isinstance(main, list):
        raise ProjectEditError("fabric.mod.json main entrypoints must be a list.")
    existing_values = {
        item if isinstance(item, str) else item.get("value")
        for item in main
        if isinstance(item, (str, dict))
    }
    if entrypoint not in existing_values:
        main.append(entrypoint)
        operations.append(
            {
                "operation": "replace",
                "path": "src/main/resources/fabric.mod.json",
                "expected_sha256": sha256_file(info.fabric_mod_json),
                "content": json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            }
        )
    if not operations:
        return {"status": "UNCHANGED", "path": str(path)}
    return TransactionalSourcePatcher(info.root).apply(operations)


def ensure_client_entrypoint(
    info: FabricProjectInfo,
    *,
    entrypoint: str,
) -> dict[str, Any]:
    raw = json.loads(info.fabric_mod_json.read_text(encoding="utf-8"))
    entrypoints = raw.setdefault("entrypoints", {})
    if not isinstance(entrypoints, dict):
        raise ProjectEditError("fabric.mod.json entrypoints must be an object.")
    client = entrypoints.setdefault("client", [])
    if not isinstance(client, list):
        raise ProjectEditError("fabric.mod.json client entrypoints must be a list.")
    existing = {
        item if isinstance(item, str) else item.get("value")
        for item in client
        if isinstance(item, (str, dict))
    }
    if entrypoint in existing:
        return {"status": "UNCHANGED", "path": str(info.fabric_mod_json)}
    client.append(entrypoint)
    return TransactionalSourcePatcher(info.root).apply(
        [
            {
                "operation": "replace",
                "path": "src/main/resources/fabric.mod.json",
                "expected_sha256": sha256_file(info.fabric_mod_json),
                "content": json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
            }
        ]
    )


def ensure_dependency(
    info: FabricProjectInfo,
    *,
    repository_block: str,
    dependency_line: str,
    marker: str,
) -> dict[str, Any]:
    build = info.root / "build.gradle"
    if not build.is_file() or build.is_symlink():
        raise ProjectEditError("build.gradle is missing.")
    text = build.read_text(encoding="utf-8")
    changed = text
    repository_marker = f"// MMM:{marker}:repository"
    if repository_block.strip() and repository_marker not in changed:
        match = re.search(r"repositories\s*\{", changed)
        if not match:
            changed += (
                "\nrepositories {\n    "
                + repository_marker
                + "\n"
                + _indent(repository_block.strip(), 4)
                + "\n}\n"
            )
        else:
            position = match.end()
            changed = (
                changed[:position]
                + "\n    "
                + repository_marker
                + "\n"
                + _indent(repository_block.strip(), 4)
                + changed[position:]
            )
    dependency_marker = f"// MMM:{marker}:dependency"
    if dependency_marker not in changed:
        match = re.search(r"dependencies\s*\{", changed)
        if not match:
            raise ProjectEditError("Could not locate dependencies block.")
        position = match.end()
        changed = (
            changed[:position]
            + "\n    "
            + dependency_marker
            + "\n    "
            + dependency_line.strip()
            + changed[position:]
        )
    if changed == text:
        return {"status": "UNCHANGED", "path": str(build)}
    return TransactionalSourcePatcher(info.root).apply(
        [
            {
                "operation": "replace",
                "path": "build.gradle",
                "expected_sha256": sha256_file(build),
                "content": changed,
            }
        ]
    )


def write_text_files(
    info: FabricProjectInfo,
    files: dict[str, str],
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for relative, content in sorted(files.items()):
        path = info.root / relative
        if path.exists():
            if not path.is_file() or path.is_symlink():
                raise ProjectEditError(
                    f"Generated target is not a regular file: {relative}"
                )
            if not replace_existing:
                current = path.read_text(encoding="utf-8")
                if current == content:
                    continue
                raise ProjectEditError(
                    f"Generated target already exists: {relative}"
                )
            operations.append(
                {
                    "operation": "replace",
                    "path": relative,
                    "expected_sha256": sha256_file(path),
                    "content": content,
                }
            )
        else:
            operations.append(
                {"operation": "create", "path": relative, "content": content}
            )
    if not operations:
        return {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "UNCHANGED",
            "operations": [],
        }
    return TransactionalSourcePatcher(info.root).apply(operations)


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line else line for line in text.splitlines()
    )
