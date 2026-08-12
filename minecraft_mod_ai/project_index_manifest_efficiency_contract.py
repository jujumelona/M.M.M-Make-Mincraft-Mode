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
    try:
        resolved = candidate.resolve()
        resolved.relative_to(target_parent.resolve())
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or resolved.is_symlink():
        return None
    return resolved


def _is_complete_current_snapshot(
    module: Any,
    *,
    existing: dict[str, Any],
    target: Path,
    version: str,
    expected_parts: int,
) -> bool:
    if existing.get("schema_version") != "mmm/project-index-v2":
        return False
    if existing.get("sha256") != f"sha256:{version}":
        return False
    if existing.get("shard_size") != module._MANIFEST_SHARD_SIZE:
        return False
    records = existing.get("parts")
    if not isinstance(records, list) or len(records) != expected_parts:
        return False
    expected_root = (target.parent / f"{target.stem}-parts" / version).resolve()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return False
        part = _safe_existing_part(target.parent, record.get("path"))
        if part is None:
            return False
        try:
            part.relative_to(expected_root)
        except ValueError:
            return False
        if part.name != f"part-{index:08d}.json":
            return False
        if not isinstance(record.get("sha256"), str):
            return False
    return True


def install(project_index_module: Any) -> None:
    """Keep immutable ProjectIndex snapshots while removing repeated full I/O.

    ProjectIndex is committed after every successful generation node. Rewriting every
    256-file shard and then reading every shard back only to hash bytes that were just
    serialized turns incremental generation into O(nodes * indexed_files) metadata
    I/O. This wrapper preserves the v2 on-disk schema and immutable version folders,
    but:

    * returns immediately when the current committed receipt is already complete;
    * hashes the exact serialized bytes in memory instead of rereading new shards;
    * reuses unchanged previous shards with verified hard links when possible;
    * falls back to an ordinary write on filesystems without hard-link support.

    The previous shard is hashed before linking, so a stale/corrupt stored receipt can
    never be used as authority for new snapshot contents.
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

        if _is_complete_current_snapshot(
            project_index_module,
            existing=existing,
            target=target,
            version=version,
            expected_parts=expected_parts,
        ):
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
                        # Stored metadata alone is not trusted. Verify the immutable
                        # source shard before sharing its inode with the new snapshot.
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
            except BaseException:
                if temporary_root.is_dir():
                    import shutil

                    shutil.rmtree(temporary_root)
                raise

        part_records: list[dict[str, str]] = []
        for name, _raw, digest in rendered_parts:
            part = version_root / name
            if not part.is_file() or part.is_symlink():
                # Preserve the base implementation's fail-closed behavior if an
                # immutable snapshot directory was externally damaged.
                raise OSError(f"Project index shard is missing: {part}")
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
        return target

    write_manifest._mmm_incremental_manifest_io = True  # type: ignore[attr-defined]
    write_manifest.__wrapped__ = current  # type: ignore[attr-defined]
    cls.write_manifest = write_manifest


__all__ = ["install"]
