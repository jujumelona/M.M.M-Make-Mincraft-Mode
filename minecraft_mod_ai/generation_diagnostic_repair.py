from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

from .mutation_authority import MutationAuthority, MutationAuthorityMode

_MAX_REPAIR_SOURCE_BYTES = 12 * 1024


def _diagnostic_paths(error: Mapping[str, Any], root: Path) -> Iterator[Path]:
    for key in ("uri", "path", "file"):
        raw = str(error.get(key) or "").strip()
        if not raw:
            continue
        try:
            if raw.startswith("file:"):
                parsed = urlsplit(raw)
                if (
                    parsed.netloc not in ("", "localhost")
                    or parsed.query
                    or parsed.fragment
                ):
                    continue
                candidate = Path(url2pathname(parsed.path))
            else:
                candidate = Path(raw)
            candidate = (root / candidate).resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        yield candidate


def read_authorized_diagnostic_source(
    errors: Sequence[Mapping[str, Any]],
    workspace_root: Any,
    authority: MutationAuthority | None,
    *,
    preferred_path: str | None = None,
) -> dict[str, Any] | None:
    """Bind a verifier's file location to fresh source within existing authority.

    Diagnostic text never grants permission or supplies the source. Read a bounded
    complete snapshot from the runtime's own workspace; preserve its exact bytes
    for edit preconditions. Large files remain on the existing retrieval path.
    """
    if (
        not workspace_root
        or authority is None
        or authority.mode is not MutationAuthorityMode.BOUNDED_ROOTS
    ):
        return None
    root = Path(str(workspace_root)).resolve()
    preferred = str(preferred_path or "").replace("\\", "/").strip()
    while preferred.startswith("./"):
        preferred = preferred[2:]
    for error in errors:
        for candidate in _diagnostic_paths(error, root):
            try:
                relative = candidate.relative_to(root).as_posix()
                if candidate.suffix.casefold() not in {".java", ".kt"}:
                    continue
                if preferred and relative != preferred:
                    continue
                if not authority.authorizes(relative, operation="replace_exact"):
                    continue
                with candidate.open("rb") as stream:
                    data = stream.read(_MAX_REPAIR_SOURCE_BYTES + 1)
                if not data or len(data) > _MAX_REPAIR_SOURCE_BYTES:
                    continue
                source = data.decode("utf-8")
            except (OSError, UnicodeError, ValueError):
                continue
            return {
                "path": relative,
                "source": source,
                "sha256": hashlib.sha256(data).hexdigest(),
                "diagnostics": [
                    {**row, "path": relative}
                    for row in errors
                    if candidate in _diagnostic_paths(row, root)
                ],
            }
    return None
