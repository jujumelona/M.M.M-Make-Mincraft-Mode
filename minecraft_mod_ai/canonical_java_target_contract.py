from __future__ import annotations

"""Fail-closed source/target checks for model-generated canonical Java candidates.

This deliberately validates only facts that the HOST can prove without guessing
Minecraft method signatures. Semantic/JDT/Gradle validation still owns correctness.
"""

import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any


_LEGACY_YARN_26 = re.compile(
    r"\bnet\.minecraft\.(?:"
    r"client\.gui\.(?:screen|widget)(?:\.|\b)|"
    r"text\.(?:Text|MutableText)\b|"
    r"util\.(?:Identifier|WorldSavePath)\b|"
    r"server\.network\.(?:ServerPlayerEntity|ServerPlayNetworkHandler)\b|"
    r"entity\.effect\.(?:StatusEffect|StatusEffectInstance)\b"
    r")"
)
_REGISTRATION_SCREEN_BASE = re.compile(
    r"\bpublic\s+(?:final\s+)?class\s+\w+\s+extends\s+"
    r"(?:net\.minecraft\.client\.gui\.screens\.)?(?:Screen|HandledScreen)\b"
)


def assert_canonical_java_target(source: str, spec: Mapping[str, Any]) -> None:
    """Reject known epoch/source-set violations *before* writing candidate files."""
    if not isinstance(source, str):
        raise ValueError("CANONICAL_SOURCE_TEXT_REQUIRED")
    bindings = spec.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("CANONICAL_SOURCE_TARGET_BINDINGS_REQUIRED")
    version = str(bindings.get("minecraft_version") or "")
    side = str(spec.get("side") or "").upper()
    leaf = str(spec.get("leaf_id") or "")
    target_path = str(spec.get("target_path") or "").replace("\\", "/")
    if not version.startswith("26."):
        return

    # These are Yarn symbols, not 26.1+ Mojang names. No string-based remap:
    # signatures and Fabric API registration conventions change with the epoch.
    # Strip comments and Java string literals, leaving only candidate code.
    without_comments = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    without_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', without_comments)
    match = _LEGACY_YARN_26.search(without_strings)
    if match is not None:
        raise ValueError(
            f"CANONICAL_MOJANG_API_REQUIRED: leaf={leaf} target={version}; "
            f"obsolete Yarn symbol {match.group(0)!r}; reject candidate before Gradle"
        )

    if side == "CLIENT":
        if not target_path.startswith("src/client/java/"):
            raise ValueError(
                f"CANONICAL_CLIENT_SOURCE_SET_REQUIRED: leaf={leaf}; "
                f"target={target_path!r} must be under src/client/java/"
            )
        if leaf == "minecraft/screen/registration" and _REGISTRATION_SCREEN_BASE.search(
            without_strings
        ):
            raise ValueError(
                "CANONICAL_SCREEN_REGISTRATION_CLASS_INVALID: registration "
                "initializer must not be a Screen subclass; implement screen "
                "UI separately with verified constructor and client API"
            )


__all__ = ["assert_canonical_java_target"]
