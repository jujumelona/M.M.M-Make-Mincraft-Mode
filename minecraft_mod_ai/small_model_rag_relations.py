from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][\w.]*)\s*;")
_IMPORT = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z_][\w.]*)\s*;")
_RESOURCE = re.compile(r"\b[a-z0-9_.-]+:[a-z0-9_./-]+\b")
_SUFFIXES = frozenset({".java", ".json", ".gradle", ".kts", ".properties", ".toml", ".yaml", ".yml", ".mcfunction", ".snbt"})
_IGNORE = frozenset({".git", ".gradle", "build", "run", "node_modules", ".minecraft_ai"})


def _files(roots: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for root in roots:
        candidates = (root,) if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in _SUFFIXES:
                continue
            try:
                relative = path.relative_to(root if root.is_dir() else root.parent)
                if any(part in _IGNORE for part in relative.parts) or path.stat().st_size > 2 * 1024 * 1024:
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
        rest = list(parts[index + 2:])
        rest[-1] = Path(rest[-1]).stem
        result.append(f"{namespace}:{'/'.join(rest)}")
        if len(rest) >= 2:
            result.append(f"{namespace}:{'/'.join(rest[1:])}")
    return tuple(dict.fromkeys(result))


def derive_relations(roots: Sequence[Path]) -> list[dict[str, str]]:
    texts: dict[Path, str] = {}
    classes: dict[str, Path] = {}
    resources: dict[str, Path] = {}
    for path in _files(roots):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        texts[path] = text
        if path.suffix.lower() == ".java":
            match = _PACKAGE.search(text)
            package = match.group(1) if match else ""
            classes[f"{package}.{path.stem}" if package else path.stem] = path
        for alias in _aliases(path):
            resources.setdefault(alias.casefold(), path)
    edges: set[tuple[str, str, str]] = set()
    for source, text in texts.items():
        if source.suffix.lower() == ".java":
            for imported in _IMPORT.findall(text):
                target = classes.get(imported)
                if target is None and "." in imported:
                    target = classes.get(imported.rsplit(".", 1)[0])
                if target is not None and target != source:
                    edges.add((str(source), str(target), "java_import"))
        for resource_id in _RESOURCE.findall(text.casefold()):
            target = resources.get(resource_id)
            if target is not None and target != source:
                edges.add((str(source), str(target), "resource_ref"))
    return [
        {"source": source, "target": target, "kind": kind}
        for source, target, kind in sorted(edges)[:50_000]
    ]


__all__ = ["derive_relations"]
