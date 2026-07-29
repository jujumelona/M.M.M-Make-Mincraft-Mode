from __future__ import annotations

from pathlib import Path


def config_path(name: str) -> Path:
    """Resolve repository config in editable installs and package config in wheels."""
    repository = Path(__file__).resolve().parents[1] / "config" / name
    if repository.is_file():
        return repository
    packaged = Path(__file__).resolve().parent / "config" / name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(f"MMM config not found: {name}")
