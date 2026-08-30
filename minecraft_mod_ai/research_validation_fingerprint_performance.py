from __future__ import annotations

"""Stat-validated content digests for validation fingerprint hot paths.

The canonical fingerprint owners live in ``validation_execution_contract``. This
module only caches file-content digests after verifying stable filesystem metadata,
so correctness never depends on runtime function replacement order.
"""

import hashlib
import os
import stat
import threading
from collections import OrderedDict
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


def content_digest(path: str | Path) -> bytes:
    """Return SHA-256 bytes, caching only a stable regular-file snapshot."""

    candidate = Path(path)
    for _attempt in range(3):
        before = _stat_key(candidate)
        with _CACHE_LOCK:
            cached = _DIGEST_CACHE.get(before)
            if cached is not None:
                _DIGEST_CACHE.move_to_end(before)
                return cached

        value = _hash_file_uncached(candidate)
        after = _stat_key(candidate)
        if before != after:
            continue

        with _CACHE_LOCK:
            _DIGEST_CACHE[after] = value
            _DIGEST_CACHE.move_to_end(after)
            while len(_DIGEST_CACHE) > _cache_limit():
                _DIGEST_CACHE.popitem(last=False)
        return value
    raise OSError(f"Validation fingerprint input changed while hashing: {candidate}")


def harden(validation_module: Any) -> None:
    """Mark canonical fingerprint owners; never replace their implementations."""

    setattr(validation_module.project_build_fingerprint, _MARKER, True)
    setattr(validation_module._java_fingerprint, _MARKER, True)


def clear_digest_cache() -> None:
    with _CACHE_LOCK:
        _DIGEST_CACHE.clear()


__all__ = ["clear_digest_cache", "content_digest", "harden"]
