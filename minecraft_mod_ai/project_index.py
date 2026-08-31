from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .scale_policy import ScalePolicy

_TEXT_SUFFIXES = {
    ".java",
    ".json",
    ".mcmeta",
    ".mcfunction",
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
_SPECIAL_TEXT_NAMES = {
    "build.gradle",
    "settings.gradle",
    "gradle.properties",
    "fabric.mod.json",
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
_RANK_CACHE_MAX_ENTRIES = 32
_SOURCE_CACHE_MAX_ENTRIES = 32
_SOURCE_CACHE_MAX_BYTES = 8 * 1024 * 1024
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


@dataclass(frozen=True)
class _SourceCacheEntry:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    verified_sha256: str | None
    content: bytes


class ProjectIndex:
    """Whole-project metadata index with native incremental update and manifest I/O."""

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
        self._manifest_receipt_cache: dict[str, Any] | None = None
        self._ranked_files_cache: dict[
            tuple[tuple[str, ...], tuple[str, ...]], tuple[IndexedFile, ...]
        ] = {}
        self._manifest_fast_receipts: set[tuple[str, str]] = set()
        self._manifest_shard_cache: dict[int, tuple[bytes, str]] = {}
        self._source_cache: dict[str, _SourceCacheEntry] = {}
        self._source_cache_bytes = 0

    def _scan(self) -> tuple[IndexedFile, ...]:
        indexed: list[IndexedFile] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.root)
            item = self._indexed_file(relative.as_posix(), path)
            if item is not None:
                indexed.append(item)
        return tuple(indexed)

    def _indexed_file(self, normalized: str, path: Path | None) -> IndexedFile | None:
        if path is None or not path.is_file() or path.is_symlink():
            return None
        relative = Path(normalized)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            return None
        suffix = path.suffix.lower()
        if suffix not in _TEXT_SUFFIXES and path.name not in _SPECIAL_TEXT_NAMES:
            return None
        size = path.stat().st_size
        if size > self.policy.max_single_file_bytes:
            tokens: tuple[str, ...] = ()
            digest = self._sha256(path)
        else:
            raw = path.read_bytes()
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            text = raw.decode("utf-8", errors="replace")
            tokens = tuple(sorted({token.lower() for token in _TOKEN.findall(text)}))
        return IndexedFile(
            path=normalized,
            size_bytes=size,
            sha256=digest,
            suffix=suffix,
            tokens=tokens,
        )

    def _resolve_touched(self, raw_path: str | Path) -> tuple[str, Path | None] | None:
        raw = Path(raw_path)
        if raw.is_absolute():
            try:
                relative = raw.relative_to(self.root)
            except ValueError:
                return None
        else:
            relative = raw
        normalized = relative.as_posix()
        if not normalized or normalized == ".":
            return None
        candidate = self.root / relative
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, ValueError):
            return normalized, None
        return normalized, candidate

    @staticmethod
    def _find_position(files: list[IndexedFile], path: str) -> tuple[int, bool]:
        lo = 0
        hi = len(files)
        while lo < hi:
            mid = (lo + hi) // 2
            if files[mid].path < path:
                lo = mid + 1
            else:
                hi = mid
        return lo, lo < len(files) and files[lo].path == path

    def _invalidate_after_update(
        self,
        *,
        dirty_shards: set[int],
        structural_from: int | None,
    ) -> None:
        self._manifest_receipt_cache = None
        self._ranked_files_cache.clear()
        self._manifest_fast_receipts.clear()
        if structural_from is not None:
            for shard in tuple(self._manifest_shard_cache):
                if shard >= structural_from:
                    self._manifest_shard_cache.pop(shard, None)
        for shard in dirty_shards:
            self._manifest_shard_cache.pop(shard, None)

    def update_files(self, touched_paths: Iterable[str | Path]) -> None:
        """Update only touched sorted entries and invalidate only affected caches."""

        files = list(self.files)
        dirty_shards: set[int] = set()
        structural_from: int | None = None
        changed = False

        for raw_path in touched_paths:
            resolved = self._resolve_touched(raw_path)
            if resolved is None:
                continue
            normalized, path = resolved
            self._drop_source_cache(normalized)
            position, existed = self._find_position(files, normalized)
            before = files[position] if existed else None
            item = self._indexed_file(normalized, path)

            if item is None:
                if not existed:
                    self._by_path.pop(normalized, None)
                    continue
                self._by_path.pop(normalized, None)
                files.pop(position)
                changed = True
                shard = position // _MANIFEST_SHARD_SIZE
                structural_from = (
                    shard if structural_from is None else min(structural_from, shard)
                )
                continue

            if existed and before == item:
                self._by_path[normalized] = item
                continue

            self._by_path[normalized] = item
            changed = True
            shard = position // _MANIFEST_SHARD_SIZE
            if existed:
                files[position] = item
                dirty_shards.add(shard)
            else:
                files.insert(position, item)
                structural_from = (
                    shard if structural_from is None else min(structural_from, shard)
                )

        if not changed:
            return
        self.files = tuple(files)
        self._invalidate_after_update(
            dirty_shards=dirty_shards,
            structural_from=structural_from,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()

    @staticmethod
    def _source_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )

    def _drop_source_cache(self, normalized: str) -> None:
        cached = self._source_cache.pop(normalized, None)
        if cached is None:
            return
        self._source_cache_bytes = max(
            0,
            self._source_cache_bytes - len(cached.content),
        )

    def _remember_source_bytes(
        self,
        normalized: str,
        *,
        stat_result: os.stat_result,
        verified_sha256: str | None,
        raw: bytes,
    ) -> None:
        self._drop_source_cache(normalized)
        if len(raw) > _SOURCE_CACHE_MAX_BYTES:
            return
        while self._source_cache and (
            len(self._source_cache) >= _SOURCE_CACHE_MAX_ENTRIES
            or self._source_cache_bytes + len(raw) > _SOURCE_CACHE_MAX_BYTES
        ):
            self._drop_source_cache(next(iter(self._source_cache)))
        self._source_cache[normalized] = _SourceCacheEntry(
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            ctime_ns=stat_result.st_ctime_ns,
            verified_sha256=verified_sha256,
            content=raw,
        )
        self._source_cache_bytes += len(raw)

    def _read_indexed_bytes(
        self,
        item: IndexedFile,
        *,
        verify_sha256: bool = False,
    ) -> bytes:
        """Read an indexed text file once and safely reuse its immutable snapshot.

        Cache hits are accepted only while the path still resolves to the same file
        identity/size/mtime tuple. Paginated source reads additionally retain the
        existing index SHA-256 guard so a stale cursor can never hide an edit.
        """

        path = self.root / item.path
        try:
            current_stat = path.stat()
        except OSError:
            self._drop_source_cache(item.path)
            raise
        current_signature = self._source_signature(current_stat)
        cached = self._source_cache.get(item.path)
        if cached is not None:
            cached_signature = (
                cached.device,
                cached.inode,
                cached.size_bytes,
                cached.mtime_ns,
                cached.ctime_ns,
            )
            if cached_signature == current_signature:
                verified = cached.verified_sha256
                if verify_sha256 and verified is None:
                    verified = "sha256:" + hashlib.sha256(cached.content).hexdigest()
                    if verified != item.sha256:
                        self._drop_source_cache(item.path)
                        raise ValueError(
                            "Project source changed after its context index was built: "
                            f"{item.path}"
                        )
                    cached = _SourceCacheEntry(
                        device=cached.device,
                        inode=cached.inode,
                        size_bytes=cached.size_bytes,
                        mtime_ns=cached.mtime_ns,
                        ctime_ns=cached.ctime_ns,
                        verified_sha256=verified,
                        content=cached.content,
                    )
                elif verify_sha256 and verified != item.sha256:
                    self._drop_source_cache(item.path)
                    raise ValueError(
                        "Project source changed after its context index was built: "
                        f"{item.path}"
                    )
                self._source_cache.pop(item.path, None)
                self._source_cache[item.path] = cached
                return cached.content
            self._drop_source_cache(item.path)

        with path.open("rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            raw = stream.read()
        verified = None
        if verify_sha256:
            verified = "sha256:" + hashlib.sha256(raw).hexdigest()
            if verified != item.sha256:
                raise ValueError(
                    "Project source changed after its context index was built: "
                    f"{item.path}"
                )

        try:
            after_stat = path.stat()
        except OSError:
            return raw
        if self._source_signature(after_stat) == self._source_signature(opened_stat):
            self._remember_source_bytes(
                item.path,
                stat_result=opened_stat,
                verified_sha256=verified,
                raw=raw,
            )
        return raw

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
            "total_text_bytes": sum(item.size_bytes for item in self.files),
            "files": [item.to_dict() for item in self.files],
        }

    @staticmethod
    def _copy_receipt(value: dict[str, Any]) -> dict[str, Any]:
        copied = dict(value)
        suffix_counts = value.get("suffix_counts")
        if isinstance(suffix_counts, dict):
            copied["suffix_counts"] = dict(suffix_counts)
        return copied

    def manifest_receipt(self) -> dict[str, Any]:
        """Return a cached fixed-size stable commitment to the complete source tree."""

        cached = self._manifest_receipt_cache
        if cached is not None:
            return self._copy_receipt(cached)

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
        value = {
            "schema_version": "mmm/project-index-receipt-v2",
            "file_count": len(self.files),
            "total_text_bytes": total_bytes,
            "sha256": "sha256:" + digest.hexdigest(),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "expanded_manifest": ".minecraft_ai/project-index.json",
        }
        self._manifest_receipt_cache = self._copy_receipt(value)
        return self._copy_receipt(value)

    def select(
        self,
        *,
        query: str | Iterable[str],
        diagnostic_paths: Iterable[str] = (),
        byte_budget: int | None = None,
    ) -> dict[str, Any]:
        budget = self.policy.model_context_bytes if byte_budget is None else byte_budget
        if type(budget) is not int or budget < 1:
            raise ValueError("byte_budget must be a positive integer")
        query_text = query if isinstance(query, str) else " ".join(str(value) for value in query)
        query_tokens = {token.lower() for token in _TOKEN.findall(query_text)}
        explicit = {self._normalize_path(value) for value in diagnostic_paths if value}

        selected: list[dict[str, Any]] = []
        consumed = 0
        for item in self._ranked_files(query_tokens=query_tokens, explicit=explicit):
            remaining = budget - consumed
            if remaining <= 0:
                break
            if item.size_bytes > self.policy.max_single_file_bytes:
                continue
            text = self._read_indexed_bytes(item).decode(
                "utf-8", errors="replace"
            )
            text = text.replace("\r\n", "\n").replace("\r", "\n")
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
        """Return one bounded page from the complete relevance-ordered source."""

        budget = self.policy.model_context_bytes if byte_budget is None else byte_budget
        if type(budget) is not int or budget < _MIN_PROJECT_CONTEXT_PAGE_BYTES:
            raise ValueError(
                "Project context page byte_budget must be an integer >= "
                f"{_MIN_PROJECT_CONTEXT_PAGE_BYTES}."
            )
        query_text = query if isinstance(query, str) else " ".join(str(value) for value in query)
        query_tokens = {token.lower() for token in _TOKEN.findall(query_text)}
        explicit = {self._normalize_path(value) for value in diagnostic_paths if value}
        ranked = [
            item
            for item in self._ranked_files(
                query_tokens=query_tokens,
                explicit=explicit,
            )
            if item.size_bytes <= self.policy.max_single_file_bytes
        ]
        query_sha256 = "sha256:" + hashlib.sha256(query_text.encode("utf-8")).hexdigest()
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
            current = self._read_indexed_bytes(item, verify_sha256=True)
            normalized = current.decode("utf-8", errors="replace").encode("utf-8")
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
        key = (tuple(sorted(query_tokens)), tuple(sorted(explicit)))
        cached = self._ranked_files_cache.get(key)
        if cached is not None:
            return list(cached)

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
        value = tuple(item for _, _, _, item in scored)
        if len(self._ranked_files_cache) >= _RANK_CACHE_MAX_ENTRIES:
            self._ranked_files_cache.pop(next(iter(self._ranked_files_cache)))
        self._ranked_files_cache[key] = value
        return list(value)

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
                return candidate.resolve().relative_to(self.root).as_posix()
            except ValueError:
                return ""
        normalized = candidate.as_posix()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    @staticmethod
    def _part_bytes(index: int, members: tuple[IndexedFile, ...]) -> bytes:
        payload = {
            "schema_version": "mmm/project-index-part-v2",
            "part": index,
            "files": [item.to_dict() for item in members],
        }
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _load_index(target: Path) -> dict[str, Any]:
        if not target.is_file() or target.is_symlink():
            return {}
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _safe_existing_part(target_parent: Path, relative: Any) -> Path | None:
        if not isinstance(relative, str) or not relative:
            return None
        candidate = target_parent / relative
        if candidate.is_symlink():
            return None
        try:
            resolved = candidate.resolve()
            resolved.relative_to(target_parent.resolve())
        except (OSError, ValueError):
            return None
        if not resolved.is_file():
            return None
        return resolved

    def _current_snapshot_parts(
        self,
        *,
        existing: dict[str, Any],
        target: Path,
        version: str,
        expected_parts: int,
    ) -> list[tuple[Path, str]] | None:
        if existing.get("schema_version") != "mmm/project-index-v2":
            return None
        if existing.get("sha256") != f"sha256:{version}":
            return None
        if existing.get("shard_size") != _MANIFEST_SHARD_SIZE:
            return None
        records = existing.get("parts")
        if not isinstance(records, list) or len(records) != expected_parts:
            return None
        expected_root = (target.parent / f"{target.stem}-parts" / version).resolve()
        parts: list[tuple[Path, str]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                return None
            part = self._safe_existing_part(target.parent, record.get("path"))
            digest = record.get("sha256")
            if part is None or not isinstance(digest, str):
                return None
            try:
                part.relative_to(expected_root)
            except ValueError:
                return None
            if part.name != f"part-{index:08d}.json":
                return None
            parts.append((part, digest))
        return parts

    def write_manifest(self, path: str | Path | None = None) -> Path:
        """Persist only changed manifest shards and reuse validated existing shards."""

        target = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.root / ".minecraft_ai/project-index.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = self.manifest_receipt()
        version = str(receipt["sha256"]).removeprefix("sha256:")
        expected_parts = (
            len(self.files) + _MANIFEST_SHARD_SIZE - 1
        ) // _MANIFEST_SHARD_SIZE
        existing = self._load_index(target)
        current_parts = self._current_snapshot_parts(
            existing=existing,
            target=target,
            version=version,
            expected_parts=expected_parts,
        )
        fast_key = (str(target), version)

        if current_parts is not None and fast_key in self._manifest_fast_receipts:
            return target
        if current_parts is not None and all(
            self._sha256(part) == digest for part, digest in current_parts
        ):
            self._manifest_fast_receipts.add(fast_key)
            return target

        previous_by_name: dict[str, tuple[Path, str]] = {}
        records = existing.get("parts")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                source = self._safe_existing_part(target.parent, record.get("path"))
                digest = record.get("sha256")
                if source is not None and isinstance(digest, str):
                    previous_by_name[source.name] = (source, digest)

        parts_root = target.parent / f"{target.stem}-parts"
        version_root = parts_root / version
        rendered_parts: list[tuple[str, bytes, str]] = []
        for index, start in enumerate(
            range(0, len(self.files), _MANIFEST_SHARD_SIZE)
        ):
            cached = self._manifest_shard_cache.get(index)
            if cached is None:
                raw = self._part_bytes(
                    index,
                    self.files[start : start + _MANIFEST_SHARD_SIZE],
                )
                digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                self._manifest_shard_cache[index] = (raw, digest)
            else:
                raw, digest = cached
            rendered_parts.append((f"part-{index:08d}.json", raw, digest))

        created_version = False
        if not version_root.is_dir():
            temporary_root = parts_root / f".{version}.{uuid.uuid4().hex}.tmp"
            temporary_root.mkdir(parents=True, exist_ok=False)
            try:
                for name, raw, digest in rendered_parts:
                    destination = temporary_root / name
                    previous = previous_by_name.get(name)
                    linked = False
                    if previous is not None and previous[1] == digest:
                        source = previous[0]
                        if self._sha256(source) == digest:
                            try:
                                os.link(source, destination)
                                linked = True
                            except OSError:
                                linked = False
                    if not linked:
                        destination.write_bytes(raw)
                parts_root.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_root, version_root)
                created_version = True
            except BaseException:
                if temporary_root.is_dir():
                    import shutil

                    shutil.rmtree(temporary_root)
                raise

        part_records: list[dict[str, str]] = []
        for name, _raw, digest in rendered_parts:
            part = version_root / name
            if not part.is_file() or part.is_symlink():
                raise OSError(f"Project index shard is missing: {part}")
            if not created_version and self._sha256(part) != digest:
                raise OSError(f"Project index shard digest mismatch: {part}")
            part_records.append(
                {
                    "path": part.relative_to(target.parent).as_posix(),
                    "sha256": digest,
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
        self._manifest_fast_receipts.add(fast_key)
        return target


ProjectIndex.manifest_receipt._mmm_cached_manifest_receipt = True  # type: ignore[attr-defined]
ProjectIndex._ranked_files._mmm_cached_relevance_order = True  # type: ignore[attr-defined]
ProjectIndex.update_files._mmm_incremental_sorted_update = True  # type: ignore[attr-defined]
ProjectIndex.write_manifest._mmm_incremental_manifest_io = True  # type: ignore[attr-defined]


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
        if query_tokens & {token.lower() for token in _TOKEN.findall(line)}
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
