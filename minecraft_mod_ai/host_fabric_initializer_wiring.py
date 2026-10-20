"""Wire graph-proven generated Java into Fabric startup, never from prompt text.

A Java class compiling successfully does not cause Fabric to load it. Register
actual ModInitializer/ClientModInitializer owners in fabric.mod.json. For a
completed canonical registry leaf with a public static register() method, emit
a tiny independently compiled ModInitializer bootstrap calling that method.
This operation runs only after artifact materialization and graph validation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from collections.abc import Sequence

from .host_prerequisite_api import java_api_declarations
from .project_write_lock import project_write_lock


_MAIN = "net.fabricmc.api.ModInitializer"
_CLIENT = "net.fabricmc.api.ClientModInitializer"
_REGISTRATION = re.compile(r"^\s*public\s+static\s+void\s+register\s*\(\s*\)\s*;\s*$")
_INTERFACE = re.compile(r"\b(?:implements|,)\s+(?:net\.fabricmc\.api\.)?(ClientModInitializer|ModInitializer)\b")
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


def _admitted_java(root: Path, relative: str):
    if (not isinstance(relative, str)
            or not relative.startswith(("src/main/java/", "src/client/java/"))
            or not relative.endswith(".java") or Path(relative).is_absolute()
            or ".." in Path(relative).parts):
        raise ValueError("FABRIC_INIT_JAVA_PATH_UNVERIFIED")
    path = root
    for part in Path(relative).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("FABRIC_INIT_JAVA_SYMLINK")
    actual = path.resolve()
    if not actual.is_relative_to(root) or not actual.is_file():
        raise ValueError("FABRIC_INIT_JAVA_MISSING: " + relative)
    return actual


def _java_owner(path: Path, source: str, relative: str):
    api = java_api_declarations(source, path=relative)
    owners = [
        row for row in api["types"]
        if row["owner"].rsplit(".", 1)[-1] == path.stem
        and re.search(
            r"\b(?:public\s+)?(?:final\s+)?class\s+" + re.escape(path.stem) + r"\b",
            row["declaration"],
        )
    ]
    if len(owners) != 1:
        raise ValueError("FABRIC_INIT_COMPILED_OWNER_MISMATCH: " + relative)
    owner = owners[0]["owner"]
    source_path = (relative.removeprefix("src/main/java/")
                   if relative.startswith("src/main/java/")
                   else relative.removeprefix("src/client/java/"))
    expected_package = source_path.rsplit("/", 1)[0].replace("/", ".")
    if owner.rpartition(".")[0] != expected_package:
        raise ValueError("FABRIC_INIT_OWNER_PACKAGE_MISMATCH: " + relative)
    return owner, owners[0]


def _entrypoint_kind(source: str, declaration: str) -> str | None:
    # Resolve only Fabric's real loader interfaces, with an explicit import
    # or fully qualified name, not an unrelated custom ModInitializer class.
    for field, short, qualified in (
        ("client", "ClientModInitializer", _CLIENT),
        ("main", "ModInitializer", _MAIN),
    ):
        if not re.search(r"\b" + short + r"\b", declaration):
            continue
        if (qualified in declaration or
                re.search(r"(?m)^\s*import\s+" + re.escape(qualified) + r"\s*;", source)):
            return field
    return None


def wire_completed_fabric_java(
    project_root: str | Path,
    approved: Sequence[tuple[str, str]],
) -> dict:
    """Add verified startup obligations; cannot silently rewrite user classes.

    approved=(materialized relative Java path, canonical leaf) tuples. The
    caller must pass only successfully published and validated graph jobs.
    Empty/missing scaffold is not claimed as verified startup.
    """
    root = Path(project_root).resolve()
    meta = root / "src/main/resources/fabric.mod.json"
    if not approved or not meta.is_file():
        return {"status": "NOT_APPLICABLE", "wired": []}
    with project_write_lock(root):
        if meta.is_symlink():
            raise ValueError("FABRIC_INIT_METADATA_SYMLINK")
        data = json.loads(meta.read_text(encoding="utf-8"))
        entrypoints = data.get("entrypoints")
        if not isinstance(entrypoints, dict):
            raise ValueError("FABRIC_INIT_METADATA_ENTRYPOINTS_INVALID")
        mutations: list[tuple[str, str, str | None]] = []
        for relative, leaf in sorted(set(approved)):
            source_path = _admitted_java(root, relative)
            source = source_path.read_text(encoding="utf-8")
            owner, declaration = _java_owner(source_path, source, relative)
            kind = _entrypoint_kind(source, declaration["declaration"])
            if kind:
                mutations.append((kind, owner, None))
            if isinstance(leaf, str) and leaf.endswith("/registry"):
                methods = [method for method in declaration["members"]
                           if _REGISTRATION.fullmatch(method)]
                if len(methods) != 1:
                    raise ValueError("FABRIC_INIT_REGISTRY_ENTRYPOINT_MISSING: " + owner)
                package = owner.rpartition(".")[0]
                if not all(_IDENTIFIER.fullmatch(part) for part in package.split(".")):
                    raise ValueError("FABRIC_INIT_REGISTRY_PACKAGE_INVALID")
                bootstrap = "MmmRegistryBootstrap" + hashlib.sha256(
                    owner.encode("utf-8")).hexdigest()[:16]
                bootstrap_owner = package + "." + bootstrap
                bootstrap_source = (
                    "package " + package + ";\n\n"
                    "public final class " + bootstrap
                    + " implements net.fabricmc.api.ModInitializer {\n"
                    "    @Override public void onInitialize() {\n"
                    "        " + owner + ".register();\n"
                    "    }\n}\n"
                )
                bootstrap_path = root / "src/main/java" / Path(*package.split(".")) / (bootstrap + ".java")
                if bootstrap_path.is_symlink():
                    raise ValueError("FABRIC_INIT_BOOTSTRAP_SYMLINK")
                if bootstrap_path.exists() and bootstrap_path.read_text(encoding="utf-8") != bootstrap_source:
                    raise ValueError("FABRIC_INIT_BOOTSTRAP_CONFLICT: " + bootstrap_owner)
                mutations.append(("main", bootstrap_owner, bootstrap_source))
        for kind, owner, source in mutations:
            collection = entrypoints.setdefault(kind, [])
            if not isinstance(collection, list):
                raise ValueError("FABRIC_INIT_ENTRYPOINT_KIND_INVALID: " + kind)
            existing = {
                str(row.get("value")) if isinstance(row, dict) else str(row)
                for row in collection
            }
            if source is not None:
                package = owner.rpartition(".")[0]
                filename = owner.rsplit(".", 1)[-1] + ".java"
                target = root / "src/main/java" / Path(*package.split(".")) / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_text(source, encoding="utf-8")
            if owner not in existing:
                if source is not None and kind == "main":
                    # Keep the template's canonical main entrypoint first,
                    # then initialize registries before generated gameplay
                    # listeners. This also holds across separate graph shards.
                    position = 1 if collection else 0
                    while (position < len(collection)
                           and "MmmRegistryBootstrap" in str(collection[position])):
                        position += 1
                    collection.insert(position, owner)
                else:
                    collection.append(owner)
        meta.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"status": "PASS", "wired": [
            {"entrypoint": kind, "owner": owner} for kind, owner, _ in mutations
        ]}
