from __future__ import annotations

import hashlib
from functools import wraps
from pathlib import Path
from typing import Any

_RANK_CACHE_MAX_ENTRIES = 32


def _part_bytes(module: Any, *, index: int, members: Any) -> bytes:
    payload = {
        "schema_version": "mmm/project-index-part-v2",
        "part": index,
        "files": [item.to_dict() for item in members],
    }
    return (
        module.json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_index(module: Any, target: Path) -> dict[str, Any]:
    if not target.is_file() or target.is_symlink():
        return {}
    try:
        value = module.json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, module.json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
    module: Any,
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
    if existing.get("shard_size") != module._MANIFEST_SHARD_SIZE:
        return None
    records = existing.get("parts")
    if not isinstance(records, list) or len(records) != expected_parts:
        return None
    expected_root = (target.parent / f"{target.stem}-parts" / version).resolve()
    parts: list[tuple[Path, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return None
        part = _safe_existing_part(target.parent, record.get("path"))
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


def _fast_receipts(index: Any) -> set[tuple[str, str]]:
    value = getattr(index, "_mmm_manifest_fast_receipts", None)
    if isinstance(value, set):
        return value
    value = set()
    index._mmm_manifest_fast_receipts = value
    return value


def _shard_cache(index: Any) -> dict[int, tuple[bytes, str]]:
    value = getattr(index, "_mmm_manifest_shard_cache", None)
    if isinstance(value, dict):
        return value
    value = {}
    index._mmm_manifest_shard_cache = value
    return value


def _rank_cache(index: Any) -> dict[tuple[tuple[str, ...], tuple[str, ...]], tuple[Any, ...]]:
    value = getattr(index, "_mmm_ranked_files_cache", None)
    if isinstance(value, dict):
        return value
    value = {}
    index._mmm_ranked_files_cache = value
    return value


def _copy_receipt(value: dict[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    suffix_counts = value.get("suffix_counts")
    if isinstance(suffix_counts, dict):
        copied["suffix_counts"] = dict(suffix_counts)
    return copied


def _install_receipt_cache(project_index_module: Any) -> None:
    cls = project_index_module.ProjectIndex
    current = cls.manifest_receipt
    if getattr(current, "_mmm_cached_manifest_receipt", False):
        return

    @wraps(current)
    def manifest_receipt(self: Any) -> dict[str, Any]:
        cached = getattr(self, "_mmm_manifest_receipt_cache", None)
        if isinstance(cached, dict):
            return _copy_receipt(cached)
        value = current(self)
        self._mmm_manifest_receipt_cache = _copy_receipt(value)
        return _copy_receipt(value)

    manifest_receipt._mmm_cached_manifest_receipt = True  # type: ignore[attr-defined]
    manifest_receipt.__wrapped__ = current  # type: ignore[attr-defined]
    cls.manifest_receipt = manifest_receipt


def _install_rank_cache(project_index_module: Any) -> None:
    cls = project_index_module.ProjectIndex
    current = cls._ranked_files
    if getattr(current, "_mmm_cached_relevance_order", False):
        return

    @wraps(current)
    def ranked_files(self: Any, *, query_tokens: set[str], explicit: set[str]) -> list[Any]:
        key = (tuple(sorted(query_tokens)), tuple(sorted(explicit)))
        cache = _rank_cache(self)
        cached = cache.get(key)
        if cached is None:
            cached = tuple(
                current(
                    self,
                    query_tokens=query_tokens,
                    explicit=explicit,
                )
            )
            if len(cache) >= _RANK_CACHE_MAX_ENTRIES:
                cache.pop(next(iter(cache)))
            cache[key] = cached
        return list(cached)

    ranked_files._mmm_cached_relevance_order = True  # type: ignore[attr-defined]
    ranked_files.__wrapped__ = current  # type: ignore[attr-defined]
    cls._ranked_files = ranked_files


def _find_position(files: list[Any], path: str) -> tuple[int, bool]:
    lo = 0
    hi = len(files)
    while lo < hi:
        mid = (lo + hi) // 2
        if files[mid].path < path:
            lo = mid + 1
        else:
            hi = mid
    return lo, lo < len(files) and files[lo].path == path


def _resolve_touched(index: Any, raw_path: str | Path) -> tuple[str, Path | None] | None:
    raw = Path(raw_path)
    root = index.root
    if raw.is_absolute():
        try:
            relative = raw.relative_to(root)
        except ValueError:
            return None
    else:
        relative = raw
    normalized = relative.as_posix()
    if not normalized or normalized == ".":
        return None
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return normalized, None
    return normalized, candidate


def _indexed_file(module: Any, index: Any, normalized: str, path: Path | None) -> Any | None:
    if path is None or not path.is_file() or path.is_symlink():
        return None
    relative = Path(normalized)
    if any(part in module._IGNORED_PARTS for part in relative.parts):
        return None
    suffix = path.suffix.lower()
    if suffix not in module._TEXT_SUFFIXES and path.name not in {
        "build.gradle",
        "settings.gradle",
        "gradle.properties",
        "fabric.mod.json",
    }:
        return None
    size = path.stat().st_size
    if size > index.policy.max_single_file_bytes:
        tokens: tuple[str, ...] = ()
        digest = index._sha256(path)
    else:
        raw = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8", errors="replace")
        tokens = tuple(
            sorted({token.lower() for token in module._TOKEN.findall(text)})
        )
    return module.IndexedFile(
        path=normalized,
        size_bytes=size,
        sha256=digest,
        suffix=suffix,
        tokens=tokens,
    )


def _invalidate_after_update(
    index: Any,
    *,
    dirty_shards: set[int],
    structural_from: int | None,
) -> None:
    for attribute in ("_mmm_manifest_receipt_cache", "_mmm_ranked_files_cache"):
        try:
            delattr(index, attribute)
        except AttributeError:
            pass
    _fast_receipts(index).clear()
    cache = _shard_cache(index)
    if structural_from is not None:
        for shard in tuple(cache):
            if shard >= structural_from:
                cache.pop(shard, None)
    for shard in dirty_shards:
        cache.pop(shard, None)


def _install_incremental_update_files(project_index_module: Any) -> None:
    cls = project_index_module.ProjectIndex
    current = cls.update_files
    if getattr(current, "_mmm_incremental_sorted_update", False):
        return

    @wraps(current)
    def update_files(self: Any, touched_paths: Any) -> None:
        files = list(self.files)
        by_path = self._by_path
        shard_size = project_index_module._MANIFEST_SHARD_SIZE
        dirty_shards: set[int] = set()
        structural_from: int | None = None
        changed = False

        for raw_path in touched_paths:
            resolved = _resolve_touched(self, raw_path)
            if resolved is None:
                continue
            normalized, path = resolved
            position, existed = _find_position(files, normalized)
            before = files[position] if existed else None
            item = _indexed_file(
                project_index_module,
                self,
                normalized,
                path,
            )
            if item is None:
                if not existed:
                    by_path.pop(normalized, None)
                    continue
                by_path.pop(normalized, None)
                files.pop(position)
                changed = True
                shard = position // shard_size
                structural_from = (
                    shard if structural_from is None else min(structural_from, shard)
                )
                continue

            if existed and before == item:
                by_path[normalized] = item
                continue

            by_path[normalized] = item
            changed = True
            shard = position // shard_size
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
        _invalidate_after_update(
            self,
            dirty_shards=dirty_shards,
            structural_from=structural_from,
        )

    update_files._mmm_incremental_sorted_update = True  # type: ignore[attr-defined]
    update_files.__wrapped__ = current  # type: ignore[attr-defined]
    cls.update_files = update_files


def install(project_index_module: Any) -> None:
    """Keep ProjectIndex incremental in CPU work, disk I/O and integrity checks."""

    _install_receipt_cache(project_index_module)
    _install_rank_cache(project_index_module)
    _install_incremental_update_files(project_index_module)

    cls = project_index_module.ProjectIndex
    current = cls.write_manifest
    if getattr(current, "_mmm_incremental_manifest_io", False):
        return

    @wraps(current)
    def write_manifest(self: Any, path: str | Path | None = None) -> Path:
        target = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.root / ".minecraft_ai/project-index.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = self.manifest_receipt()
        version = str(receipt["sha256"]).removeprefix("sha256:")
        shard_size = project_index_module._MANIFEST_SHARD_SIZE
        expected_parts = (len(self.files) + shard_size - 1) // shard_size
        existing = _load_index(project_index_module, target)
        current_parts = _current_snapshot_parts(
            project_index_module,
            existing=existing,
            target=target,
            version=version,
            expected_parts=expected_parts,
        )
        fast_key = (str(target), version)
        fast_receipts = _fast_receipts(self)

        if current_parts is not None and fast_key in fast_receipts:
            return target
        if current_parts is not None:
            if all(self._sha256(part) == digest for part, digest in current_parts):
                fast_receipts.add(fast_key)
                return target

        previous_by_name: dict[str, tuple[Path, str]] = {}
        records = existing.get("parts")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                source = _safe_existing_part(target.parent, record.get("path"))
                digest = record.get("sha256")
                if source is not None and isinstance(digest, str):
                    previous_by_name[source.name] = (source, digest)

        parts_root = target.parent / f"{target.stem}-parts"
        version_root = parts_root / version
        shard_cache = _shard_cache(self)
        rendered_parts: list[tuple[str, bytes, str]] = []
        for index, start in enumerate(range(0, len(self.files), shard_size)):
            cached = shard_cache.get(index)
            if cached is None:
                raw = _part_bytes(
                    project_index_module,
                    index=index,
                    members=self.files[start : start + shard_size],
                )
                digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                shard_cache[index] = (raw, digest)
            else:
                raw, digest = cached
            rendered_parts.append((f"part-{index:08d}.json", raw, digest))

        created_version = False
        if not version_root.is_dir():
            temporary_root = parts_root / (
                f".{version}.{project_index_module.uuid.uuid4().hex}.tmp"
            )
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
                                project_index_module.os.link(source, destination)
                                linked = True
                            except OSError:
                                linked = False
                    if not linked:
                        destination.write_bytes(raw)
                parts_root.mkdir(parents=True, exist_ok=True)
                project_index_module.os.replace(temporary_root, version_root)
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
            "shard_size": shard_size,
            "parts": part_records,
        }
        temporary_target = target.with_name(
            f".{target.name}.{project_index_module.uuid.uuid4().hex}.tmp"
        )
        temporary_target.write_text(
            project_index_module.json.dumps(
                index_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        project_index_module.os.replace(temporary_target, target)
        fast_receipts.add(fast_key)
        return target

    write_manifest._mmm_incremental_manifest_io = True  # type: ignore[attr-defined]
    write_manifest.__wrapped__ = current  # type: ignore[attr-defined]
    cls.write_manifest = write_manifest


__all__ = ["install"]