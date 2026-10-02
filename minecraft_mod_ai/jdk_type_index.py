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
def _javap_public_output(home_text: str, fqcn: str) -> str:
    home = Path(home_text)
    javap = _javap_executable(home)
    if not javap:
        return ""
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
        return ""
    return completed.stdout if completed.returncode == 0 else ""


@lru_cache(maxsize=512)
def _public_type(home_text: str, fqcn: str) -> bool:
    return _PUBLIC_DECLARATION.search(
        _javap_public_output(home_text, fqcn)
    ) is not None


def _parameter_arity(parameter_text: str) -> tuple[int, bool]:
    text = str(parameter_text or "").strip()
    if not text:
        return 0, False
    depth = 0
    count = 1
    for char in text:
        if char in "<([{":
            depth += 1
        elif char in ">)]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            count += 1
    return count, text.rstrip().endswith("...")


def _constructor_shapes_from_javap(
    text: str,
    fqcn: str,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    simple = fqcn.rsplit(".", 1)[-1]
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line.startswith("public ") or not line.endswith(";"):
            continue
        match = re.match(r"public\s+([^\s(]+)\((.*)\);$", line)
        if match is None:
            continue
        owner = match.group(1)
        if owner not in {fqcn, simple} and not owner.endswith("." + simple):
            continue
        arity, varargs = _parameter_arity(match.group(2))
        rows.append({"arity": arity, "varargs": varargs})
    return tuple(rows)


def public_jdk_constructor_shapes(value: str) -> tuple[dict[str, object], ...]:
    """Return public constructor arity/varargs facts for one installed-JDK type."""

    fqcn = str(value or "").strip()
    if not fqcn.startswith(("java.", "javax.")):
        return ()
    home = _java_home()
    if home is None or not _public_type(str(home), fqcn):
        return ()
    return _constructor_shapes_from_javap(
        _javap_public_output(str(home), fqcn),
        fqcn,
    )


def _erase_generic_type(value: str) -> str:
    text = str(value or "").strip()
    out: list[str] = []
    depth = 0
    for char in text:
        if char == "<":
            depth += 1
            continue
        if char == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(char)
    return "".join(out).strip()


def _static_factory_shapes_from_javap(
    text: str,
    fqcn: str,
) -> tuple[dict[str, object], ...]:
    """Return public static methods that construct/return the same JDK type."""

    rows: list[dict[str, object]] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line.startswith("public static ") or not line.endswith(";"):
            continue
        match = re.match(
            r"public\s+static\s+"
            r"(?:(?:final|synchronized|native|strictfp)\s+)*"
            r"(?:<[^;]+?>\s+)?"
            r"(?P<return>[^\s(]+)\s+"
            r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)"
            r"\((?P<parameters>.*)\);$",
            line,
        )
        if match is None:
            continue
        return_type = _erase_generic_type(match.group("return"))
        if return_type != fqcn:
            continue
        arity, varargs = _parameter_arity(match.group("parameters"))
        rows.append(
            {
                "name": match.group("name"),
                "arity": arity,
                "varargs": varargs,
                "return_type": return_type,
            }
        )
    return tuple(rows)


def public_jdk_static_factory_shapes(value: str) -> tuple[dict[str, object], ...]:
    """Return same-type public static factory shapes for one installed-JDK type."""

    fqcn = str(value or "").strip()
    if not fqcn.startswith(("java.", "javax.")):
        return ()
    home = _java_home()
    if home is None or not _public_type(str(home), fqcn):
        return ()
    return _static_factory_shapes_from_javap(
        _javap_public_output(str(home), fqcn),
        fqcn,
    )


def _method_shapes_from_javap(
    text: str,
    fqcn: str,
    method_name: str = "",
) -> tuple[dict[str, object], ...]:
    """Return public method name/static/arity facts from javap output."""

    rows: list[dict[str, object]] = []
    simple = fqcn.rsplit(".", 1)[-1]
    wanted = str(method_name or "").strip()
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line.startswith("public ") or not line.endswith(";") or "(" not in line:
            continue
        signature = line[:-1]
        head, parameters = signature.split("(", 1)
        parameters = parameters.rsplit(")", 1)[0]
        name_match = re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*$", head)
        if name_match is None:
            continue
        name = name_match.group(1)
        constructor_heads = {
            f"public {fqcn}",
            f"public {simple}",
        }
        if head.strip() in constructor_heads:
            continue
        if wanted and name != wanted:
            continue
        arity, varargs = _parameter_arity(parameters)
        rows.append(
            {
                "name": name,
                "static": bool(re.search(r"\bstatic\b", head)),
                "arity": arity,
                "varargs": varargs,
            }
        )
    return tuple(rows)


def public_jdk_method_shapes(
    value: str,
    method_name: str,
) -> tuple[dict[str, object], ...]:
    """Return public method shapes for one exact installed-JDK type."""

    fqcn = str(value or "").strip()
    name = str(method_name or "").strip()
    if not fqcn.startswith(("java.", "javax.")) or not name:
        return ()
    home = _java_home()
    if home is None or not _public_type(str(home), fqcn):
        return ()
    return _method_shapes_from_javap(
        _javap_public_output(str(home), fqcn),
        fqcn,
        name,
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
        # This index contains top-level classes only. Do not collapse a nested
        # owner (Map.Entry, Thread.State, etc.) to an unrelated top-level name.
        if "$" in raw or any(part[:1].isupper() for part in raw.split(".")[:-1]):
            return raw
        simple = raw.rsplit(".", 1)[-1]
        if raw in index.get(simple, ()):
            # An existing but inaccessible type must fail admission as itself;
            # accessibility is not permission to change the selected type.
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
    "public_jdk_constructor_shapes",
    "public_jdk_static_factory_shapes",
    "public_jdk_method_shapes",
    "public_jdk_type_candidates",
]
