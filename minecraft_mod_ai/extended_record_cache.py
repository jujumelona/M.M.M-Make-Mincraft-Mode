from __future__ import annotations

import copy
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

_RECORD_CACHE_LOCK = threading.RLock()
_RECORD_CACHE: OrderedDict[
    str,
    dict[str, tuple[tuple[int, int, int], dict[str, Any]]],
] = OrderedDict()
_RECORD_CACHE_LIMIT = 16


def _record_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)


def read_cached_directory_records(
    directory: Path,
    expected: int,
    *,
    error_type: type[Exception],
) -> list[dict[str, Any]]:
    directory_key = str(directory)
    with _RECORD_CACHE_LOCK:
        cached_records = dict(_RECORD_CACHE.get(directory_key, {}))

    refreshed: dict[str, tuple[tuple[int, int, int], dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if not path.is_file() or path.is_symlink():
            raise error_type("Extended module record is unsafe.")
        signature = _record_signature(path)
        path_key = str(path)
        cached = cached_records.get(path_key)
        if cached is not None and cached[0] == signature:
            item = cached[1]
        else:
            item = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(item, dict)
                or not item.get("module_id")
                or path.stem != str(item["module_id"])
            ):
                raise error_type("Extended module record is invalid.")
        refreshed[path_key] = (signature, item)
        records.append(copy.deepcopy(item))

    if len(records) != expected:
        raise error_type("Extended module directory count does not match.")

    with _RECORD_CACHE_LOCK:
        _RECORD_CACHE[directory_key] = refreshed
        _RECORD_CACHE.move_to_end(directory_key)
        while len(_RECORD_CACHE) > _RECORD_CACHE_LIMIT:
            _RECORD_CACHE.popitem(last=False)
    return records


__all__ = [
    "_RECORD_CACHE",
    "_RECORD_CACHE_LIMIT",
    "_RECORD_CACHE_LOCK",
    "read_cached_directory_records",
]
