from __future__ import annotations

from pathlib import Path

_CONFIG_ROOT = (Path(__file__).resolve().parent / "config").resolve(strict=False)


def config_path(name: str) -> Path:
    """Resolve a configuration file from the packaged canonical config directory."""
    candidate = (_CONFIG_ROOT / name).resolve(strict=False)
    try:
        candidate.relative_to(_CONFIG_ROOT)
    except ValueError as exc:
        raise ValueError(f"config path escapes package root: {name}") from exc
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"MMM config not found: {name}")
