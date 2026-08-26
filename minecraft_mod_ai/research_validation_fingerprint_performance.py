from __future__ import annotations

"""Stat-validated content fingerprint reuse for validation/JDT hot paths.

Gradle itself reuses a prior file fingerprint when size and last-modified metadata are
unchanged. MMM previously re-read every build-relevant byte before it could discover
that the Gradle/JDT result was already cached. This layer applies the same conservative
snapshot principle one level earlier: metadata is always revalidated, unchanged regular
files reuse only their content digest, and any metadata mutation forces a fresh read.
"""

import hashlib
import os
import stat
import threading
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_MARKER = "_mmm_stat_validated_validation_fingerprint_v1"
_CACHE_LOCK = threading.RLock()
_DIGEST_CACHE: OrderedDict[tuple[str, int, int, int, int, int], bytes] = OrderedDict()
_DEFAULT_CACHE_LIMIT = 8192


def _cache_limit() -> int:
    raw = os.environ.get("MMM_VALIDATION_DIGEST_CACHE_FILES", "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_CACHE_LIMIT
    except ValueError:
        value = _DEFAULT_CACHE_LIMIT
    return max(128, min(65536, value))


def _stat_key(path: Path) -> tuple[str, int, int, int, int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise OSError(f"Validation fingerprint input is not a regular file: {path}")
    return (
        str(path.absolute()),
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _hash_file_uncached(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _content_digest(path: Path) -> bytes:
    """Return SHA-256 bytes while refusing to cache a concurrently changing input."""

    path = Path(path)
    for _attempt in range(3):
        before = _stat_key(path)
        with _CACHE_LOCK:
            cached = _DIGEST_CACHE.get(before)
            if cached is not None:
                _DIGEST_CACHE.move_to_end(before)
                return cached

        value = _hash_file_uncached(path)
        after = _stat_key(path)
        if before != after:
            continue

        with _CACHE_LOCK:
            _DIGEST_CACHE[after] = value
            _DIGEST_CACHE.move_to_end(after)
            while len(_DIGEST_CACHE) > _cache_limit():
                _DIGEST_CACHE.popitem(last=False)
        return value
    raise OSError(f"Validation fingerprint input changed while hashing: {path}")


def _iter_build_inputs(validation_module: Any, root: Path) -> tuple[tuple[str, Path], ...]:
    values: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if validation_module._is_build_input(relative):
            values.append((relative, path))
    values.sort(key=lambda item: item[0])
    return tuple(values)


def _project_fingerprint(validation_module: Any, project_root: str | Path) -> str:
    """Merkle-style exact-content identity without re-reading unchanged file bodies."""

    root = Path(project_root).expanduser().resolve()
    digest = hashlib.sha256()
    digest.update(b"mmm/build-input-fingerprint-v2\0")
    for relative, path in _iter_build_inputs(validation_module, root):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_content_digest(path))
        digest.update(b"\0")
    return digest.hexdigest()


def _java_fingerprint(
    validation_module: Any,
    project_root: str | Path,
    relative_files: Iterable[str] | None,
) -> tuple[str, tuple[str, ...]]:
    root = Path(project_root).expanduser().resolve()
    if relative_files is None:
        paths = sorted(
            (
                path
                for path in root.rglob("*.java")
                if path.is_file() and not path.is_symlink()
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        relative = tuple(path.relative_to(root).as_posix() for path in paths)
    else:
        relative = tuple(
            sorted(set(str(value).replace("\\", "/") for value in relative_files))
        )
        paths = [root / value for value in relative]

    digest = hashlib.sha256()
    digest.update(b"mmm/java-validation-fingerprint-v2\0")
    for config_name in ("build.gradle", "settings.gradle", "gradle.properties"):
        config = root / config_name
        if config.is_file() and not config.is_symlink():
            digest.update(config_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_content_digest(config))
            digest.update(b"\0")
    for rel, path in zip(relative, paths, strict=True):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_content_digest(path))
        digest.update(b"\0")
    return digest.hexdigest(), relative


def harden(validation_module: Any) -> None:
    """Replace only cache-key construction; build/JDT verification remains authoritative."""

    current_project = validation_module.project_build_fingerprint
    if not getattr(current_project, _MARKER, False):
        def project_build_fingerprint(project_root: str | Path) -> str:
            return _project_fingerprint(validation_module, project_root)

        setattr(project_build_fingerprint, _MARKER, True)
        project_build_fingerprint.__wrapped__ = current_project  # type: ignore[attr-defined]
        validation_module.project_build_fingerprint = project_build_fingerprint

    current_java = validation_module._java_fingerprint
    if not getattr(current_java, _MARKER, False):
        def java_fingerprint(
            project_root: str | Path,
            relative_files: Iterable[str] | None,
        ) -> tuple[str, tuple[str, ...]]:
            return _java_fingerprint(validation_module, project_root, relative_files)

        setattr(java_fingerprint, _MARKER, True)
        java_fingerprint.__wrapped__ = current_java  # type: ignore[attr-defined]
        validation_module._java_fingerprint = java_fingerprint


def clear_digest_cache() -> None:
    with _CACHE_LOCK:
        _DIGEST_CACHE.clear()


__all__ = ["clear_digest_cache", "harden"]
