from __future__ import annotations

"""Fail-closed source/target checks for model-generated canonical Java candidates.

This deliberately validates only facts that the HOST can prove without guessing
Minecraft method signatures. Semantic/JDT/Gradle validation still owns correctness.
"""

import re
from collections.abc import Mapping
from pathlib import Path


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



def assert_canonical_client_source_set(project_root: str | Path, spec: Mapping[str, Any]) -> None:
    """Require the official split-client Gradle contract before a 26.1 model call."""
    bindings = spec.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("CANONICAL_SOURCE_TARGET_BINDINGS_REQUIRED")
    version = str(bindings.get("minecraft_version") or "")
    if not version.startswith("26.") or str(spec.get("side") or "").upper() != "CLIENT":
        return
    root = Path(project_root).expanduser().resolve()
    build = root / "build.gradle"
    if not build.is_file() or build.is_symlink():
        raise ValueError("CANONICAL_CLIENT_BUILD_GRADLE_REQUIRED")
    script = build.read_text(encoding="utf-8")
    if ("splitEnvironmentSourceSets" not in script or
            "sourceSets.client" not in script):
        raise ValueError(
            "CANONICAL_CLIENT_GRADLE_SOURCE_SET_MISSING: Minecraft 26.1+ "
            "GUI requires loom.splitEnvironmentSourceSets() and "
            "the client sourceSet in loom.mods before source generation"
        )


def retire_owned_legacy_client_candidate(
    project_root: str | Path, spec: Mapping[str, Any]
) -> str | None:
    """Discard only the exact former main-source location of a moved GUI candidate.

    A successful new client candidate may otherwise coexist with stale invalid
    Yarn source in main on resume and fail every later compileJava invocation.
    Never remove arbitrary user files or an unproven candidate identity.
    """
    bindings = spec.get("bindings")
    if not isinstance(bindings, Mapping):
        return None
    if not str(bindings.get("minecraft_version") or "").startswith("26."):
        return None
    if str(spec.get("side") or "").upper() != "CLIENT":
        return None
    current = str(spec.get("target_path") or "").replace("\\", "/")
    if not current.startswith("src/client/java/") or "/client/generated/" not in current:
        return None
    name = str(bindings.get("class_name") or "")
    package = str(bindings.get("package_name") or "")
    if not name or not package:
        raise ValueError("CANONICAL_CLIENT_LEGACY_IDENTITY_MISSING")
    old_relative = current.replace("src/client/java/", "src/main/java/", 1)
    root = Path(project_root).expanduser().resolve()
    old = root / old_relative
    if not old.exists():
        return None
    if old.is_symlink() or not old.is_file() or not old.resolve().is_relative_to(root):
        raise ValueError("CANONICAL_CLIENT_LEGACY_PATH_UNSAFE")
    source = old.read_text(encoding="utf-8")
    package_match = re.search(
        r"(?m)^\s*package\s+([A-Za-z0-9_.]+)\s*;", source
    )
    class_match = re.search(
        r"\b(?:public\s+)?(?:final\s+)?class\s+([A-Za-z0-9_]+)\b", source
    )
    if (package_match is None or package_match.group(1) != package
            or class_match is None or class_match.group(1) != name):
        raise ValueError(
            "CANONICAL_CLIENT_LEGACY_SOURCE_CONFLICT: refusing to delete "
            f"unowned file {old_relative}"
        )
    old.unlink()
    return old_relative


__all__ = ["assert_canonical_java_target", "assert_canonical_client_source_set", "retire_owned_legacy_client_candidate"]
