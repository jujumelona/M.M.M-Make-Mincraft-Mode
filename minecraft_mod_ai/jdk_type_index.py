"""Runtime JDK type authority backed by the installed JDK image.

This module never downloads metadata. It indexes the exact JDK selected by the
runtime and uses javap only to confirm public top-level types. Generation code can
therefore canonicalize standard-library type names without maintaining an ever-growing
handwritten package table.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess


_CLASS_ROW = re.compile(r"^\s*(?P<path>(?:java|javax)/[A-Za-z0-9_$/]+)\.class\s*$")
_PUBLIC_DECLARATION = re.compile(
    r"(?m)^public\s+(?:(?:final|abstract|sealed|non-sealed)\s+)*"
    r"(?:class|interface|enum|record|@interface)\s+"
)


def _java_home() -> Path | None:
    for variable in ("MMM_PROJECT_JAVA_HOME", "JAVA_HOME"):
        configured = os.environ.get(variable, "").strip()
        if not configured:
            continue
        candidate = Path(configured).expanduser().resolve()
        if (candidate / "lib" / "modules").is_file():
            return candidate

    java = shutil.which("java")
    if not java:
        return None
    candidate = Path(java).resolve().parent.parent
    return candidate if (candidate / "lib" / "modules").is_file() else None


def _jimage_executable(home: Path) -> str | None:
    name = "jimage.exe" if os.name == "nt" else "jimage"
    bundled = home / "bin" / name
    if bundled.is_file():
        return str(bundled)
    return shutil.which("jimage")


def _javap_executable(home: Path) -> str | None:
    name = "javap.exe" if os.name == "nt" else "javap"
    bundled = home / "bin" / name
    if bundled.is_file():
        return str(bundled)
    return shutil.which("javap")


def _parse_jimage_type_index(text: str) -> dict[str, tuple[str, ...]]:
    rows: dict[str, set[str]] = {}
    for raw in str(text or "").splitlines():
        match = _CLASS_ROW.match(raw)
        if match is None:
            continue
        path = match.group("path")
        if "$" in path:
            continue
        fqcn = path.replace("/", ".")
        simple = fqcn.rsplit(".", 1)[-1]
        rows.setdefault(simple, set()).add(fqcn)
    return {
        simple: tuple(sorted(values))
        for simple, values in rows.items()
    }


@lru_cache(maxsize=4)
def _runtime_index(home_text: str) -> dict[str, tuple[str, ...]]:
    home = Path(home_text)
    jimage = _jimage_executable(home)
    modules = home / "lib" / "modules"
    if not jimage or not modules.is_file():
        return {}
    try:
        completed = subprocess.run(
            [jimage, "list", str(modules)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    return _parse_jimage_type_index(completed.stdout)


@lru_cache(maxsize=512)
def _public_type(home_text: str, fqcn: str) -> bool:
    home = Path(home_text)
    javap = _javap_executable(home)
    if not javap:
        return False
    try:
        completed = subprocess.run(
            [javap, "-public", fqcn],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return (
        completed.returncode == 0
        and _PUBLIC_DECLARATION.search(completed.stdout) is not None
    )


def public_jdk_type_candidates(simple_name: str) -> tuple[str, ...]:
    """Return public java./javax. top-level types with this simple name."""

    simple = str(simple_name or "").strip()
    if not simple or "." in simple or "$" in simple:
        return ()
    home = _java_home()
    if home is None:
        return ()
    candidates = _runtime_index(str(home)).get(simple, ())
    return tuple(
        fqcn
        for fqcn in candidates
        if _public_type(str(home), fqcn)
    )


def canonical_public_jdk_type(value: str) -> str:
    """Canonicalize one raw JDK type name when the installed JDK is unambiguous."""

    raw = str(value or "").strip()
    if not raw:
        return raw
    home = _java_home()
    if home is None:
        return raw

    index = _runtime_index(str(home))
    if raw.startswith(("java.", "javax.")):
        simple = raw.rsplit(".", 1)[-1]
        if raw in index.get(simple, ()) and _public_type(str(home), raw):
            return raw
        candidates = tuple(
            fqcn
            for fqcn in index.get(simple, ())
            if _public_type(str(home), fqcn)
        )
        return candidates[0] if len(candidates) == 1 else raw

    if "." in raw or "$" in raw:
        return raw
    candidates = public_jdk_type_candidates(raw)
    return candidates[0] if len(candidates) == 1 else raw


def is_public_jdk_type(value: str) -> bool:
    """Return whether one exact java./javax. FQCN is a public installed-JDK type."""

    raw = str(value or "").strip()
    if not raw.startswith(("java.", "javax.")):
        return False
    home = _java_home()
    if home is None:
        return False
    simple = raw.rsplit(".", 1)[-1]
    if raw not in _runtime_index(str(home)).get(simple, ()):
        return False
    return _public_type(str(home), raw)


__all__ = [
    "canonical_public_jdk_type",
    "is_public_jdk_type",
    "public_jdk_type_candidates",
]
