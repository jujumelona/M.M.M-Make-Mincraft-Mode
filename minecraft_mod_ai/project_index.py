from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .scale_policy import ScalePolicy


_TEXT_SUFFIXES = {
    ".java",
    ".json",
    ".mcmeta",
    ".gradle",
    ".properties",
    ".accesswidener",
    ".mixins",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
}
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,127}")
_IGNORED_PARTS = {".git", ".gradle", "build", "run", ".cache", "node_modules"}


@dataclass(frozen=True)
class IndexedFile:
    path: str
    size_bytes: int
    sha256: str
    suffix: str
    tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProjectIndex:
    """Whole-project metadata index with relevance retrieval.

    Every eligible source/resource file is indexed. A byte budget limits materialized
    content sent to a model, but files are ranked by diagnostics, symbol overlap,
    dependency references and path relevance rather than by directory order.
    """

    def __init__(self, project_root: str | Path, *, policy: ScalePolicy | None = None) -> None:
        self.root = Path(project_root).expanduser().resolve()
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError(f"Project root must be a regular directory: {self.root}")
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()
        self.files = self._scan()
        self._by_path = {item.path: item for item in self.files}

    def _scan(self) -> tuple[IndexedFile, ...]:
        indexed: list[IndexedFile] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.root)
            if any(part in _IGNORED_PARTS for part in relative.parts):
                continue
            suffix = path.suffix.lower()
            if suffix not in _TEXT_SUFFIXES and path.name not in {
                "build.gradle",
                "settings.gradle",
                "gradle.properties",
                "fabric.mod.json",
            }:
                continue
            size = path.stat().st_size
            if size > self.policy.max_single_file_bytes:
                # Keep metadata in the index while refusing to materialize unbounded data.
                tokens: tuple[str, ...] = ()
                digest = self._sha256(path)
            else:
                raw = path.read_bytes()
                digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                text = raw.decode("utf-8", errors="replace")
                tokens = tuple(sorted({token.lower() for token in _TOKEN.findall(text)}))
            indexed.append(
                IndexedFile(
                    path=relative.as_posix(),
                    size_bytes=size,
                    sha256=digest,
                    suffix=suffix,
                    tokens=tokens,
                )
            )
        return tuple(indexed)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/project-index-v1",
            "project_root": str(self.root),
            "file_count": len(self.files),
            "total_text_bytes": sum(item.size_bytes for item in self.files),
            "files": [item.to_dict() for item in self.files],
        }

    def select(
        self,
        *,
        query: str | Iterable[str],
        diagnostic_paths: Iterable[str] = (),
        byte_budget: int | None = None,
    ) -> dict[str, Any]:
        budget = byte_budget or self.policy.model_context_bytes
        if budget < 1:
            raise ValueError("byte_budget must be positive")
        query_text = query if isinstance(query, str) else " ".join(str(value) for value in query)
        query_tokens = {token.lower() for token in _TOKEN.findall(query_text)}
        explicit = {self._normalize_path(value) for value in diagnostic_paths if value}

        scored: list[tuple[int, int, str, IndexedFile]] = []
        for item in self.files:
            score = 0
            if item.path in explicit:
                score += 1_000_000
            path_tokens = {token.lower() for token in _TOKEN.findall(item.path)}
            score += 60 * len(query_tokens & path_tokens)
            score += 8 * len(query_tokens & set(item.tokens))
            if item.path in {
                "build.gradle",
                "settings.gradle",
                "gradle.properties",
                "src/main/resources/fabric.mod.json",
            }:
                score += 100
            if item.path.endswith("Mod.java") or item.path.endswith("Client.java"):
                score += 80
            scored.append((-score, item.size_bytes, item.path, item))
        scored.sort()

        selected: list[dict[str, Any]] = []
        consumed = 0
        for _, size, _, item in scored:
            if size > self.policy.max_single_file_bytes:
                continue
            if consumed + size > budget and selected:
                continue
            path = self.root / item.path
            text = path.read_text(encoding="utf-8", errors="replace")
            encoded_size = len(text.encode("utf-8"))
            if consumed + encoded_size > budget and selected:
                continue
            selected.append(
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "content": text,
                }
            )
            consumed += encoded_size
            if consumed >= budget:
                break

        return {
            "schema_version": "mmm/project-context-v1",
            "query_tokens": sorted(query_tokens),
            "indexed_file_count": len(self.files),
            "selected_file_count": len(selected),
            "selected_bytes": consumed,
            "byte_budget": budget,
            "files": selected,
        }

    def _normalize_path(self, value: str) -> str:
        raw = value.removeprefix("file://")
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(self.root).as_posix()
            except ValueError:
                return ""
        return candidate.as_posix().lstrip("./")

    def write_manifest(self, path: str | Path | None = None) -> Path:
        target = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.root / ".minecraft_ai/project-index.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target
