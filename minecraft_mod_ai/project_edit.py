from __future__ import annotations

import json
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


def inspect_fabric_project(project_root: str | Path) -> FabricProjectInfo:
    root = Path(project_root).expanduser().resolve()
    metadata = root / "src/main/resources/fabric.mod.json"
    if not metadata.is_file() or metadata.is_symlink():
        raise ProjectEditError("fabric.mod.json is missing from the generated project.")
    raw = json.loads(metadata.read_text(encoding="utf-8"))
    mod_id = raw.get("id")
    entrypoints = raw.get("entrypoints", {})
    main = entrypoints.get("main", []) if isinstance(entrypoints, dict) else []
    if not isinstance(mod_id, str) or not isinstance(main, list) or len(main) != 1:
        raise ProjectEditError("Expected exactly one Fabric main entrypoint.")
    entrypoint = main[0]
    if not isinstance(entrypoint, str) or "." not in entrypoint:
        raise ProjectEditError("Fabric main entrypoint is invalid.")
    package_name, main_class = entrypoint.rsplit(".", 1)
    main_java = root / "src/main/java" / Path(*package_name.split(".")) / f"{main_class}.java"
    if not main_java.is_file() or main_java.is_symlink():
        raise ProjectEditError(f"Main initializer source is missing: {main_java}")
    return FabricProjectInfo(
        root=root,
        mod_id=mod_id,
        main_entrypoint=entrypoint,
        package_name=package_name,
        main_class=main_class,
        main_java=main_java,
        fabric_mod_json=metadata,
    )


def ensure_main_initializer_call(
    info: FabricProjectInfo,
    *,
    import_line: str,
    call_line: str,
    marker: str,
) -> dict[str, Any]:
    text = info.main_java.read_text(encoding="utf-8")
    changed = text
    import_line = import_line.rstrip(";") + ";"
    if import_line not in changed:
        anchor = "import org.slf4j.Logger;"
        if anchor not in changed:
            raise ProjectEditError("Could not locate the import insertion anchor.")
        changed = changed.replace(anchor, import_line + "\n" + anchor, 1)
    marker_line = f"        // MMM:{marker}"
    rendered_call = marker_line + "\n        " + call_line.rstrip(";") + ";"
    if marker_line not in changed:
        anchor = "    public void onInitialize() {\n"
        if anchor not in changed:
            raise ProjectEditError("Could not locate ModInitializer.onInitialize().")
        changed = changed.replace(anchor, anchor + rendered_call + "\n", 1)
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
    if entrypoint in client:
        return {"status": "UNCHANGED", "path": str(info.fabric_mod_json)}
    client.append(entrypoint)
    relative = info.fabric_mod_json.relative_to(info.root).as_posix()
    return TransactionalSourcePatcher(info.root).apply(
        [
            {
                "operation": "replace",
                "path": relative,
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
    if not build.is_file():
        raise ProjectEditError("build.gradle is missing.")
    text = build.read_text(encoding="utf-8")
    changed = text
    repository_marker = f"// MMM:{marker}:repository"
    if repository_block.strip() and repository_marker not in changed:
        anchor = "repositories {\n"
        if anchor not in changed:
            raise ProjectEditError("Could not locate repositories block.")
        changed = changed.replace(
            anchor,
            anchor + f"    {repository_marker}\n" + _indent(repository_block.strip(), 4) + "\n",
            1,
        )
    dependency_marker = f"// MMM:{marker}:dependency"
    if dependency_marker not in changed:
        anchor = "dependencies {\n"
        if anchor not in changed:
            raise ProjectEditError("Could not locate dependencies block.")
        changed = changed.replace(
            anchor,
            anchor + f"    {dependency_marker}\n    {dependency_line.strip()}\n",
            1,
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
            if not replace_existing:
                current = path.read_text(encoding="utf-8")
                if current == content:
                    continue
                raise ProjectEditError(f"Generated target already exists: {relative}")
            operations.append(
                {
                    "operation": "replace",
                    "path": relative,
                    "expected_sha256": sha256_file(path),
                    "content": content,
                }
            )
        else:
            operations.append({"operation": "create", "path": relative, "content": content})
    if not operations:
        return {"schema_version": "mmm/source-patch-receipt-v1", "status": "UNCHANGED", "operations": []}
    return TransactionalSourcePatcher(info.root).apply(operations)


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())
