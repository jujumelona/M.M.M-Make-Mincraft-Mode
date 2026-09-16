from __future__ import annotations

"""Host-owned mutation authority shared by localization and final write guards.

Mutation authority is compiled by the host before generation. Model-visible payloads may
explain the active authority, but they can never create or widen it.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Iterable


class MutationAuthorityMode(str, Enum):
    EXACT = "exact"
    BOUNDED_ROOTS = "bounded_roots"


_DELETE_OPERATIONS = frozenset({"delete", "delete_file", "remove", "remove_file"})
_DEFAULT_BOUNDED_ROOTS = (
    "src/main/java/",
    "src/main/resources/",
    "src/test/java/",
    "src/gametest/",
)
_FORBIDDEN_TOP_LEVEL = frozenset(
    {
        ".mmm",
        ".git",
        "build",
        "config",
        "gradle",
    }
)
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_DRIVE_RE = re.compile(r"^[A-Za-z]:($|[/\\])")


class MutationAuthorityError(RuntimeError):
    """Raised when a host mutation authority cannot be formed safely."""


def canonical_mutation_path(value: Any) -> str:
    """Return a safe repository-relative POSIX path, or an empty string when unsafe."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    if _URI_RE.match(raw) or _DRIVE_RE.match(raw):
        return ""
    raw = raw.replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw or raw.startswith("/"):
        return ""

    parts = PurePosixPath(raw).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    if parts[0].casefold() in _FORBIDDEN_TOP_LEVEL:
        return ""
    if any(part.casefold() == ".mmm" for part in parts):
        return ""
    return PurePosixPath(*parts).as_posix()


def _canonical_root(value: Any) -> str:
    path = canonical_mutation_path(value)
    if not path:
        return ""
    return path.rstrip("/") + "/"


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class MutationAuthority:
    """Immutable host-owned write authority.

    EXACT authorizes only explicit paths. BOUNDED_ROOTS authorizes descendants of the
    generated source/resource roots and never authorizes delete operations.
    """

    mode: MutationAuthorityMode
    paths: tuple[str, ...] = ()
    roots: tuple[str, ...] = ()
    task_id: str = ""

    @classmethod
    def exact(cls, paths: Iterable[Any], *, task_id: str = "") -> "MutationAuthority":
        canonical = _dedupe(
            path for path in (canonical_mutation_path(item) for item in paths) if path
        )
        if not canonical:
            raise MutationAuthorityError(
                "MUTATION_AUTHORITY_EMPTY: exact authority requires at least one safe path."
            )
        return cls(
            mode=MutationAuthorityMode.EXACT,
            paths=canonical,
            task_id=str(task_id or "").strip(),
        )

    @classmethod
    def bounded_roots(
        cls,
        roots: Iterable[Any] = _DEFAULT_BOUNDED_ROOTS,
        *,
        task_id: str = "",
    ) -> "MutationAuthority":
        canonical = _dedupe(root for root in (_canonical_root(item) for item in roots) if root)
        if not canonical:
            raise MutationAuthorityError(
                "MUTATION_AUTHORITY_EMPTY: bounded authority requires at least one safe root."
            )
        allowed = set(_DEFAULT_BOUNDED_ROOTS)
        if not set(canonical).issubset(allowed):
            raise MutationAuthorityError(
                "MUTATION_AUTHORITY_ROOT_FORBIDDEN: bounded roots must stay inside generated "
                "Java/resource/test/gametest source sets."
            )
        return cls(
            mode=MutationAuthorityMode.BOUNDED_ROOTS,
            roots=canonical,
            task_id=str(task_id or "").strip(),
        )

    def authorizes(self, value: Any, *, operation: Any = "") -> bool:
        path = canonical_mutation_path(value)
        if not path:
            return False
        operation_name = str(operation or "").strip().casefold()
        if self.mode is MutationAuthorityMode.EXACT:
            return path in self.paths
        if operation_name in _DELETE_OPERATIONS or operation_name.startswith(("delete_", "remove_")):
            return False
        return any(path.startswith(root) and len(path) > len(root) for root in self.roots)

    def mutation_error(self, value: Any, *, operation: Any = "") -> str | None:
        path = canonical_mutation_path(value)
        if not path:
            return (
                "PATH_OUTSIDE_WRITABLE_SET: mutation path must be a safe repository-relative "
                "path inside host-owned authority."
            )
        operation_name = str(operation or "").strip().casefold()
        if self.mode is MutationAuthorityMode.BOUNDED_ROOTS and (
            operation_name in _DELETE_OPERATIONS
            or operation_name.startswith(("delete_", "remove_"))
        ):
            return (
                "WRITE_SCOPE_DELETE_FORBIDDEN: authored design generation may create or edit "
                f"bounded project files but may not delete {path!r}."
            )
        if self.authorizes(path, operation=operation_name):
            return None
        if self.mode is MutationAuthorityMode.EXACT:
            return (
                f"MUTATION_TARGET_DRIFT: host authority permits {list(self.paths)!r} but "
                f"mutation requested {path!r}."
            )
        return (
            "PATH_OUTSIDE_WRITABLE_SET: authored design writes are restricted to "
            f"{list(self.roots)!r}; mutation requested {path!r}."
        )


AUTHORED_DESIGN_ROOTS = _DEFAULT_BOUNDED_ROOTS


__all__ = [
    "AUTHORED_DESIGN_ROOTS",
    "MutationAuthority",
    "MutationAuthorityError",
    "MutationAuthorityMode",
    "canonical_mutation_path",
]
