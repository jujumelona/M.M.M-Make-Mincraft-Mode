from __future__ import annotations

import hashlib
from functools import wraps
from pathlib import Path
from typing import Any


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


def install(project_index_module: Any) -> None:
    """Keep immutable ProjectIndex snapshots while removing repeated full I/O.

    ProjectIndex is committed after every successful generation node. Rewriting every
    256-file shard and then reading every shard back only to hash bytes that were just
    serialized turns incremental generation into O(nodes * indexed_files) metadata
    I/O. This wrapper preserves the v2 on-disk schema and immutable version folders,
    but:

    * same-process repeated commits of the exact receipt use a zero-read fast path;
    * resumed-process snapshots are digest-verified once before entering that fast path;
    * new shard digests are computed from serialized bytes instead of disk readback;
    * unchanged previous shards are reused through verified hard links when possible;
    * filesystems without hard-link support fall back to ordinary writes.

    Stored metadata alone is never authority for an old shard: every shard reused from
    a previous snapshot is hashed first. Existing version directories encountered after
    restart are also checked against the newly rendered bytes before their receipt is
    published.
    """

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
        rendered_parts: list[tuple[str, bytes, str]] = []
        for index, start in enumerate(range(0, len(self.files), shard_size)):
            name = f"part-{index:08d}.json"
            raw = _part_bytes(
                project_index_module,
                index=index,
                members=self.files[start : start + shard_size],
            )
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            rendered_parts.append((name, raw, digest))

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
