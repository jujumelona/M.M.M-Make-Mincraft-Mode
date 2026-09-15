from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][\w.]*)\s*;")
_IMPORT = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z_][\w.]*)\s*;")
_JAVA_TYPE = re.compile(r"\b([A-Z][A-Za-z0-9_$]{1,127})\b")
_RESOURCE = re.compile(r"\b[a-z0-9_.-]+:[a-z0-9_./-]+\b")
_IDENTIFIER_PAIR = re.compile(
    r"(?:new\s+)?(?:Identifier|ResourceLocation)"
    r"(?:\.of)?\s*\(\s*[\"']([a-z0-9_.-]+)[\"']\s*,\s*"
    r"[\"']([a-z0-9_./-]+)[\"']\s*\)",
    flags=re.IGNORECASE,
)
_MIXIN_TARGET = re.compile(
    r"@Mixin\s*\(\s*(?:value\s*=\s*)?([A-Za-z_][\w.$]*)\.class",
)
_SUFFIXES = frozenset(
    {
        ".java",
        ".json",
        ".gradle",
        ".kts",
        ".properties",
        ".toml",
        ".yaml",
        ".yml",
        ".mcfunction",
        ".snbt",
    }
)
_IGNORE = frozenset(
    {".git", ".gradle", "build", "run", "node_modules", ".minecraft_ai"}
)


def _files(roots: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for root in roots:
        candidates = (root,) if root.is_file() else root.rglob("*")
        for path in candidates:
            if (
                not path.is_file()
                or path.is_symlink()
                or path.suffix.lower() not in _SUFFIXES
            ):
                continue
            try:
                relative = path.relative_to(root if root.is_dir() else root.parent)
                if (
                    any(part in _IGNORE for part in relative.parts)
                    or path.stat().st_size > 2 * 1024 * 1024
                ):
                    continue
            except (OSError, ValueError):
                continue
            result.append(path.resolve())
    return sorted(set(result), key=str)


def _aliases(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    parts = path.parts
    for marker in ("assets", "data"):
        try:
            index = parts.index(marker)
        except ValueError:
            continue
        if len(parts) <= index + 2:
            continue
        namespace = parts[index + 1]
        rest = list(parts[index + 2 :])
        rest[-1] = Path(rest[-1]).stem
        result.append(f"{namespace}:{'/'.join(rest)}")
        if len(rest) >= 2:
            result.append(f"{namespace}:{'/'.join(rest[1:])}")
    return tuple(dict.fromkeys(result))


def _json_mapping(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_java_class(
    name: str,
    classes: dict[str, Path],
    simple_classes: dict[str, set[Path]],
) -> Path | None:
    normalized = name.replace("$", ".")
    direct = classes.get(normalized)
    if direct is not None:
        return direct
    simple = normalized.rsplit(".", 1)[-1]
    matches = simple_classes.get(simple, set())
    if len(matches) == 1:
        return next(iter(matches))
    return None


def _mixin_entries(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def derive_relations(roots: Sequence[Path]) -> list[dict[str, str]]:
    texts: dict[Path, str] = {}
    classes: dict[str, Path] = {}
    simple_classes: dict[str, set[Path]] = {}
    resources: dict[str, Path] = {}
    json_by_name: dict[str, list[Path]] = {}

    for path in _files(roots):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        texts[path] = text
        if path.suffix.lower() == ".java":
            match = _PACKAGE.search(text)
            package = match.group(1) if match else ""
            qualified = f"{package}.{path.stem}" if package else path.stem
            classes[qualified] = path
            simple_classes.setdefault(path.stem, set()).add(path)
        if path.suffix.lower() == ".json":
            json_by_name.setdefault(path.name.casefold(), []).append(path)
        for alias in _aliases(path):
            resources.setdefault(alias.casefold(), path)

    edges: set[tuple[str, str, str]] = set()
    gradle_properties = [
        path for path in texts if path.name.casefold() == "gradle.properties"
    ]
    gradle_settings = [
        path
        for path in texts
        if path.name.casefold() in {"settings.gradle", "settings.gradle.kts"}
    ]
    version_catalogs = [
        path for path in texts if path.name.casefold() == "libs.versions.toml"
    ]

    for source, text in texts.items():
        suffix = source.suffix.lower()
        if suffix == ".java":
            for imported in _IMPORT.findall(text):
                target = classes.get(imported)
                if target is None and "." in imported:
                    target = classes.get(imported.rsplit(".", 1)[0])
                if target is not None and target != source:
                    edges.add((str(source), str(target), "java_import"))

            for simple_name in set(_JAVA_TYPE.findall(text)):
                target = _resolve_java_class(simple_name, classes, simple_classes)
                if target is not None and target != source:
                    edges.add((str(source), str(target), "java_type"))

            for raw_target in _MIXIN_TARGET.findall(text):
                target = _resolve_java_class(raw_target, classes, simple_classes)
                if target is not None and target != source:
                    edges.add((str(source), str(target), "mixin_target"))

            for namespace, value in _IDENTIFIER_PAIR.findall(text):
                target = resources.get(f"{namespace}:{value}".casefold())
                if target is not None and target != source:
                    edges.add((str(source), str(target), "registry_ref"))

        for resource_id in _RESOURCE.findall(text.casefold()):
            target = resources.get(resource_id)
            if target is not None and target != source:
                edges.add((str(source), str(target), "resource_ref"))

        if source.name.casefold().endswith(".mixins.json"):
            payload = _json_mapping(text)
            if payload is not None:
                package = str(payload.get("package", "")).strip()
                for key in ("mixins", "client", "server"):
                    for entry in _mixin_entries(payload.get(key)):
                        qualified = f"{package}.{entry}" if package else entry
                        target = _resolve_java_class(
                            qualified,
                            classes,
                            simple_classes,
                        )
                        if target is not None and target != source:
                            edges.add((str(source), str(target), "mixin_class"))

        if source.name.casefold() == "fabric.mod.json":
            payload = _json_mapping(text)
            mixins = payload.get("mixins") if payload is not None else None
            if isinstance(mixins, list):
                for item in mixins:
                    config = ""
                    if isinstance(item, str):
                        config = item.strip()
                    elif isinstance(item, dict):
                        config = str(item.get("config", "")).strip()
                    if not config:
                        continue
                    candidates = json_by_name.get(Path(config).name.casefold(), [])
                    if len(candidates) == 1 and candidates[0] != source:
                        edges.add(
                            (str(source), str(candidates[0]), "fabric_mixin_config")
                        )

        if source.name.casefold() in {
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
        }:
            for target in gradle_properties:
                if target != source:
                    edges.add((str(source), str(target), "gradle_properties"))
            for target in gradle_settings:
                if target != source:
                    edges.add((str(source), str(target), "gradle_settings"))
            if "libs." in text or "versioncatalog" in text.casefold():
                for target in version_catalogs:
                    if target != source:
                        edges.add(
                            (str(source), str(target), "gradle_version_catalog")
                        )

    return [
        {"source": source, "target": target, "kind": kind}
        for source, target, kind in sorted(edges)[:50_000]
    ]


__all__ = ["derive_relations"]
