from __future__ import annotations

from pathlib import Path


def read_gradle_properties(path: Path) -> dict[str, str]:
    """Read a concrete Gradle properties file without following symlinks."""

    if not path.is_file() or path.is_symlink():
        raise ValueError(f"gradle.properties is missing: {path}")
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


__all__ = ["read_gradle_properties"]
