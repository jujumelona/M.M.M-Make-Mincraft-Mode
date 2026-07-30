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
_IGNORED_PARTS = {
    ".git",
    ".gradle",
    "build",
    "run",
    ".cache",
    "node_modules",
}


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
    """Whole-project metadata index with byte-bounded relevance retrieval."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        policy: ScalePolicy | None = None,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError(
                f"Project root must be a regular directory: {self.root}"
            )
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
                tokens: tuple[str, ...] = ()
                digest = self._sha256(path)
            else:
                raw = path.read_bytes()
                digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                text = raw.decode("utf-8", errors="replace")
                tokens = tuple(
                    sorted(
                        {
                            token.lower()
                            for token in _TOKEN.findall(text)
                        }
                    )
                )
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
            for chunk in iter(
                lambda: stream.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/project-index-v1",
            "project_root": str(self.root),
            "file_count": len(self.files),
            "total_text_bytes": sum(
                item.size_bytes for item in self.files
            ),
            "files": [item.to_dict() for item in self.files],
        }

    def select(
        self,
        *,
        query: str | Iterable[str],
        diagnostic_paths: Iterable[str] = (),
        byte_budget: int | None = None,
    ) -> dict[str, Any]:
        budget = (
            self.policy.model_context_bytes
            if byte_budget is None
            else byte_budget
        )
        if type(budget) is not int or budget < 1:
            raise ValueError("byte_budget must be a positive integer")
        query_text = (
            query
            if isinstance(query, str)
            else " ".join(str(value) for value in query)
        )
        query_tokens = {
            token.lower() for token in _TOKEN.findall(query_text)
        }
        explicit = {
            self._normalize_path(value)
            for value in diagnostic_paths
            if value
        }

        scored: list[tuple[int, int, str, IndexedFile]] = []
        for item in self.files:
            score = 0
            if item.path in explicit:
                score += 1_000_000
            path_tokens = {
                token.lower() for token in _TOKEN.findall(item.path)
            }
            score += 60 * len(query_tokens & path_tokens)
            score += 8 * len(query_tokens & set(item.tokens))
            if item.path in {
                "build.gradle",
                "settings.gradle",
                "gradle.properties",
                "src/main/resources/fabric.mod.json",
            }:
                score += 100
            if item.path.endswith("Mod.java") or item.path.endswith(
                "Client.java"
            ):
                score += 80
            scored.append(
                (-score, item.size_bytes, item.path, item)
            )
        scored.sort()

        selected: list[dict[str, Any]] = []
        consumed = 0
        for _, _, _, item in scored:
            remaining = budget - consumed
            if remaining <= 0:
                break
            if item.size_bytes > self.policy.max_single_file_bytes:
                continue
            path = self.root / item.path
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            raw = text.encode("utf-8")
            truncated = False
            if len(raw) > remaining:
                if selected and item.path not in explicit:
                    continue
                text = _relevant_excerpt(
                    text,
                    query_tokens=query_tokens,
                    byte_budget=remaining,
                )
                raw = text.encode("utf-8")
                truncated = len(raw) < item.size_bytes
            if not raw:
                continue
            if len(raw) > remaining:
                raw = raw[:remaining]
                text = raw.decode("utf-8", errors="ignore")
                raw = text.encode("utf-8")
                truncated = True
            selected.append(
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "content_bytes": len(raw),
                    "truncated": truncated,
                    "content": text,
                }
            )
            consumed += len(raw)

        return {
            "schema_version": "mmm/project-context-v2",
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
                return (
                    candidate.resolve()
                    .relative_to(self.root)
                    .as_posix()
                )
            except ValueError:
                return ""
        normalized = candidate.as_posix()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def write_manifest(
        self,
        path: str | Path | None = None,
    ) -> Path:
        target = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.root / ".minecraft_ai/project-index.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.manifest(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return target


def _relevant_excerpt(
    text: str,
    *,
    query_tokens: set[str],
    byte_budget: int,
) -> str:
    if byte_budget <= 0:
        return ""
    lines = text.splitlines(keepends=True)
    matching = [
        index
        for index, line in enumerate(lines)
        if query_tokens
        & {
            token.lower() for token in _TOKEN.findall(line)
        }
    ]
    if not matching:
        raw = text.encode("utf-8")[:byte_budget]
        return raw.decode("utf-8", errors="ignore")
    chosen: set[int] = set()
    radius = 4
    for index in matching:
        chosen.update(
            range(
                max(0, index - radius),
                min(len(lines), index + radius + 1),
            )
        )
    chunks: list[str] = []
    consumed = 0
    previous = -2
    for index in sorted(chosen):
        separator = "\n...\n" if index > previous + 1 else ""
        candidate = separator + lines[index]
        size = len(candidate.encode("utf-8"))
        if consumed + size > byte_budget:
            remaining = byte_budget - consumed
            if remaining > 0:
                chunks.append(
                    candidate.encode("utf-8")[:remaining].decode(
                        "utf-8",
                        errors="ignore",
                    )
                )
            break
        chunks.append(candidate)
        consumed += size
        previous = index
    return "".join(chunks)
