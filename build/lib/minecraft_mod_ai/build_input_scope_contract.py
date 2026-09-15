from __future__ import annotations

from pathlib import Path
from typing import Any


def install(validation_module: Any) -> None:
    """Keep build fingerprints limited to bytes that can affect Gradle output."""

    original = validation_module._is_build_input
    if getattr(original, "_mmm_build_input_scope", False):
        return

    def is_build_input(relative: str) -> bool:
        normalized = str(relative).replace("\\", "/")
        path = Path(normalized)
        if path.parts and path.parts[0] == ".minecraft_ai":
            return False
        return original(normalized)

    is_build_input._mmm_build_input_scope = True
    validation_module._is_build_input = is_build_input
