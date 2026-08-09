from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
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
    # Generated receipts contain timestamps and indexes of this index. They are
    # audit metadata, not project source, and including them makes every resume
    # fingerprint change itself.
    ".minecraft_ai",
}
_MANIFEST_SHARD_SIZE = 256
_PROJECT_CONTEXT_CURSOR = re.compile(
    r"^pc1:(?P<position>[0-9]+):(?P<offset>[0-9]+):(?P<page>[0-9]+):"
    r"(?P<fingerprint>[0-9a-f]{24})$"
)
_MIN_PROJECT_CONTEXT_PAGE_BYTES = 1024


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
        """Return the legacy expanded view for explicit compatibility callers.

        Model prompts and durable checkpoint fingerprints must use
        :meth:`manifest_receipt`; expanding every path into every request creates a
        project-size context ceiling.
        """

        return {
            "schema_version": "mmm/project-index-v1",
            "project_root": str(self.root),
            "file_count": len(self.files),
            "total_text_bytes": sum(
                item.size_bytes for item in self.files
            ),
            "files": [item.to_dict() for item in self.files],
        }

    def manifest_receipt(self) -> dict[str, Any]:
        """Return a fixed-size, stable commitment to the complete source tree."""

        digest = hashlib.sha256()
        total_bytes = 0
        suffix_counts: dict[str, int] = {}
        for item in self.files:
            record = json.dumps(
                {
                    "path": item.path,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                    "suffix": item.suffix,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(record).to_bytes(8, "big"))
            digest.update(record)
            total_bytes += item.size_bytes
            suffix_counts[item.suffix or "<none>"] = (
                suffix_counts.get(item.suffix or "<none>", 0) + 1
            )
        return {
            "schema_version": "mmm/project-index-receipt-v2",
            "file_count": len(self.files),
            "total_text_bytes": total_bytes,
            "sha256": "sha256:" + digest.hexdigest(),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "expanded_manifest": ".minecraft_ai/project-index.json",
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

        selected: list[dict[str, Any]] = []
        consumed = 0
        for item in self._ranked_files(
            query_tokens=query_tokens,
            explicit=explicit,
        ):
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

    def select_page(
        self,
        *,
        query: str | Iterable[str],
        diagnostic_paths: Iterable[str] = (),
        byte_budget: int | None = None,
        cursor: str = "",
    ) -> dict[str, Any]:
        """Return one bounded page from the complete relevance-ordered source.

        ``select`` intentionally remains the compatibility one-shot retrieval API.
        This method adds an independent, host-owned cursor for generation workflows:
        every text file allowed by the explicit host file-size policy is reachable,
        and files larger than one page are continued as UTF-8 fragments instead of
        being silently abandoned.  The cursor is bound to the project receipt,
        query, explicit paths and byte budget, so it cannot be replayed against a
        different source snapshot.
        """

        budget = (
            self.policy.model_context_bytes
            if byte_budget is None
            else byte_budget
        )
        if type(budget) is not int or budget < _MIN_PROJECT_CONTEXT_PAGE_BYTES:
            raise ValueError(
                "Project context page byte_budget must be an integer >= "
                f"{_MIN_PROJECT_CONTEXT_PAGE_BYTES}."
            )
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
        ranked = [
            item
            for item in self._ranked_files(
                query_tokens=query_tokens,
                explicit=explicit,
            )
            if item.size_bytes <= self.policy.max_single_file_bytes
        ]
        query_sha256 = "sha256:" + hashlib.sha256(
            query_text.encode("utf-8")
        ).hexdigest()
        project_sha256 = self.manifest_receipt()["sha256"]
        fingerprint = self._page_fingerprint(
            project_sha256=project_sha256,
            query_sha256=query_sha256,
            diagnostic_paths=explicit,
            byte_budget=budget,
        )
        position, offset, page_index = self._decode_page_cursor(
            cursor,
            fingerprint=fingerprint,
            eligible_file_count=len(ranked),
        )
        start_position = position
        start_offset = offset
        selected: list[dict[str, Any]] = []

        while position < len(ranked):
            item = ranked[position]
            path = self.root / item.path
            current = path.read_bytes()
            current_sha256 = "sha256:" + hashlib.sha256(current).hexdigest()
            if current_sha256 != item.sha256:
                raise ValueError(
                    "Project source changed after its context index was built: "
                    f"{item.path}"
                )
            normalized = current.decode(
                "utf-8",
                errors="replace",
            ).encode("utf-8")
            if offset > len(normalized):
                raise ValueError("Project context cursor byte offset is invalid.")
            if not normalized and offset == 0:
                empty_record = self._context_file_record(
                    item=item,
                    content=b"",
                    start=0,
                    total=0,
                )
                next_position = position + 1
                candidate = self._project_context_page(
                    project_sha256=project_sha256,
                    query_sha256=query_sha256,
                    fingerprint=fingerprint,
                    indexed_file_count=len(self.files),
                    eligible_file_count=len(ranked),
                    excluded_file_count=len(self.files) - len(ranked),
                    byte_budget=budget,
                    page_index=page_index,
                    start_position=start_position,
                    start_offset=start_offset,
                    files=[*selected, empty_record],
                    next_position=next_position,
                    next_offset=0,
                )
                if _json_size(candidate) > budget:
                    if selected:
                        break
                    raise ValueError(
                        "Project context page byte_budget is too small for file metadata."
                    )
                selected.append(empty_record)
                position = next_position
                offset = 0
                continue

            remaining = len(normalized) - offset
            if remaining <= 0:
                position += 1
                offset = 0
                continue
            content_size = self._largest_fitting_fragment(
                item=item,
                normalized=normalized,
                start=offset,
                max_size=remaining,
                existing=selected,
                query_sha256=query_sha256,
                project_sha256=project_sha256,
                fingerprint=fingerprint,
                indexed_file_count=len(self.files),
                eligible_file_count=len(ranked),
                excluded_file_count=len(self.files) - len(ranked),
                byte_budget=budget,
                page_index=page_index,
                start_position=start_position,
                start_offset=start_offset,
                position=position,
            )
            if content_size < 1:
                if selected:
                    break
                raise ValueError(
                    "Project context page byte_budget is too small for a source fragment."
                )
            content = normalized[offset : offset + content_size]
            record = self._context_file_record(
                item=item,
                content=content,
                start=offset,
                total=len(normalized),
            )
            selected.append(record)
            offset += len(content)
            if offset >= len(normalized):
                position += 1
                offset = 0
            else:
                break

        result = self._project_context_page(
            project_sha256=project_sha256,
            query_sha256=query_sha256,
            fingerprint=fingerprint,
            indexed_file_count=len(self.files),
            eligible_file_count=len(ranked),
            excluded_file_count=len(self.files) - len(ranked),
            byte_budget=budget,
            page_index=page_index,
            start_position=start_position,
            start_offset=start_offset,
            files=selected,
            next_position=position,
            next_offset=offset,
        )
        if _json_size(result) > budget:
            raise AssertionError("Project context page exceeded its byte budget.")
        return result

    def _ranked_files(
        self,
        *,
        query_tokens: set[str],
        explicit: set[str],
    ) -> list[IndexedFile]:
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
            scored.append((-score, item.size_bytes, item.path, item))
        scored.sort()
        return [item for _, _, _, item in scored]

    def _page_fingerprint(
        self,
        *,
        project_sha256: str,
        query_sha256: str,
        diagnostic_paths: set[str],
        byte_budget: int,
    ) -> str:
        commitment = {
            "project_sha256": project_sha256,
            "query_sha256": query_sha256,
            "diagnostic_paths": sorted(diagnostic_paths),
            "byte_budget": byte_budget,
        }
        return hashlib.sha256(
            json.dumps(
                commitment,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]

    @staticmethod
    def _decode_page_cursor(
        cursor: str,
        *,
        fingerprint: str,
        eligible_file_count: int,
    ) -> tuple[int, int, int]:
        if not cursor:
            return 0, 0, 0
        match = _PROJECT_CONTEXT_CURSOR.fullmatch(cursor)
        if match is None or match.group("fingerprint") != fingerprint:
            raise ValueError(
                "Project context cursor does not match this source/query snapshot."
            )
        position = int(match.group("position"))
        offset = int(match.group("offset"))
        page = int(match.group("page"))
        if position > eligible_file_count or offset < 0 or page < 1:
            raise ValueError("Project context cursor position is invalid.")
        if position == eligible_file_count and offset:
            raise ValueError("Completed project context cursor has a byte offset.")
        return position, offset, page

    @staticmethod
    def _encode_page_cursor(
        *,
        position: int,
        offset: int,
        page: int,
        fingerprint: str,
    ) -> str:
        return f"pc1:{position}:{offset}:{page}:{fingerprint}"

    @staticmethod
    def _context_file_record(
        *,
        item: IndexedFile,
        content: bytes,
        start: int,
        total: int,
    ) -> dict[str, Any]:
        text = content.decode("utf-8", errors="strict")
        end = start + len(content)
        return {
            "path": item.path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "content_start_bytes": start,
            "content_end_bytes": end,
            "normalized_content_bytes": total,
            "content_complete": start == 0 and end == total,
            "content": text,
        }

    def _largest_fitting_fragment(
        self,
        *,
        item: IndexedFile,
        normalized: bytes,
        start: int,
        max_size: int,
        existing: list[dict[str, Any]],
        project_sha256: str,
        query_sha256: str,
        fingerprint: str,
        indexed_file_count: int,
        eligible_file_count: int,
        excluded_file_count: int,
        byte_budget: int,
        page_index: int,
        start_position: int,
        start_offset: int,
        position: int,
    ) -> int:
        low = 1
        high = max_size
        best = 0
        while low <= high:
            requested = (low + high) // 2
            raw = normalized[start : start + requested]
            text = raw.decode("utf-8", errors="ignore")
            content = text.encode("utf-8")
            if not content:
                low = requested + 1
                continue
            end = start + len(content)
            next_position = position + 1 if end >= len(normalized) else position
            next_offset = 0 if next_position != position else end
            record = self._context_file_record(
                item=item,
                content=content,
                start=start,
                total=len(normalized),
            )
            candidate = self._project_context_page(
                project_sha256=project_sha256,
                query_sha256=query_sha256,
                fingerprint=fingerprint,
                indexed_file_count=indexed_file_count,
                eligible_file_count=eligible_file_count,
                excluded_file_count=excluded_file_count,
                byte_budget=byte_budget,
                page_index=page_index,
                start_position=start_position,
                start_offset=start_offset,
                files=[*existing, record],
                next_position=next_position,
                next_offset=next_offset,
            )
            if _json_size(candidate) <= byte_budget:
                best = len(content)
                low = requested + 1
            else:
                high = requested - 1
        return best

    def _project_context_page(
        self,
        *,
        project_sha256: str,
        query_sha256: str,
        fingerprint: str,
        indexed_file_count: int,
        eligible_file_count: int,
        excluded_file_count: int,
        byte_budget: int,
        page_index: int,
        start_position: int,
        start_offset: int,
        files: list[dict[str, Any]],
        next_position: int,
        next_offset: int,
    ) -> dict[str, Any]:
        complete = next_position >= eligible_file_count and next_offset == 0
        next_cursor = (
            ""
            if complete
            else self._encode_page_cursor(
                position=next_position,
                offset=next_offset,
                page=page_index + 1,
                fingerprint=fingerprint,
            )
        )
        return {
            "schema_version": "mmm/project-context-page-v3",
            "project_sha256": project_sha256,
            "query_sha256": query_sha256,
            "indexed_file_count": indexed_file_count,
            "eligible_file_count": eligible_file_count,
            "excluded_by_host_file_size_policy": excluded_file_count,
            "page_index": page_index,
            "start_position": start_position,
            "start_offset": start_offset,
            "selected_file_fragment_count": len(files),
            "selected_content_bytes": sum(
                item["content_end_bytes"] - item["content_start_bytes"]
                for item in files
            ),
            "byte_budget": byte_budget,
            "complete": complete,
            "next_cursor": next_cursor,
            "files": files,
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
        receipt = self.manifest_receipt()
        version = receipt["sha256"].removeprefix("sha256:")
        parts_root = target.parent / f"{target.stem}-parts"
        version_root = parts_root / version
        if not version_root.is_dir():
            temporary_root = parts_root / f".{version}.{uuid.uuid4().hex}.tmp"
            temporary_root.mkdir(parents=True, exist_ok=False)
            try:
                for index, start in enumerate(
                    range(0, len(self.files), _MANIFEST_SHARD_SIZE)
                ):
                    members = self.files[start : start + _MANIFEST_SHARD_SIZE]
                    payload = {
                        "schema_version": "mmm/project-index-part-v2",
                        "part": index,
                        "files": [item.to_dict() for item in members],
                    }
                    (temporary_root / f"part-{index:08d}.json").write_text(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                parts_root.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_root, version_root)
            except BaseException:
                if temporary_root.is_dir():
                    import shutil

                    shutil.rmtree(temporary_root)
                raise
        part_records = []
        for part in sorted(version_root.glob("part-*.json")):
            part_records.append(
                {
                    "path": part.relative_to(target.parent).as_posix(),
                    "sha256": self._sha256(part),
                }
            )
        index_payload = {
            **receipt,
            "schema_version": "mmm/project-index-v2",
            "shard_size": _MANIFEST_SHARD_SIZE,
            "parts": part_records,
        }
        temporary_target = target.with_name(
            f".{target.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary_target.write_text(
            json.dumps(
                index_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_target, target)
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


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
