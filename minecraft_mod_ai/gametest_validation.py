from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def validate_gametest_metadata(
    validator: Any,
    root: Path,
    spec: Any,
    findings: list[Any],
    finding_type: Callable[..., Any],
    entrypoint_values: Callable[[Any], set[str]],
    class_name: Callable[[str], str],
) -> int:
    """Validate Loom's dedicated GameTest mod and source set."""

    checks = 0
    metadata_path = root / "src/gametest/resources/fabric.mod.json"
    metadata = validator._load_json(metadata_path, findings, root)
    checks += 1

    for key, expected in (
        ("id", f"{spec.mod_id}_gametest"),
        ("environment", "*"),
    ):
        checks += 1
        if metadata.get(key) != expected:
            findings.append(
                finding_type(
                    "BAD_GAMETEST_METADATA",
                    "error",
                    validator._rel(root, metadata_path),
                    f"{key} must equal {expected!r}.",
                )
            )

    gametest_class = (
        f"{spec.package_name}.{class_name(spec.mod_id)}ModGameTests"
    )
    entrypoints = metadata.get("entrypoints")
    checks += 1
    if not isinstance(entrypoints, dict):
        findings.append(
            finding_type(
                "BAD_GAMETEST_ENTRYPOINTS",
                "error",
                validator._rel(root, metadata_path),
                "GameTest entrypoints must be an object.",
            )
        )
    else:
        checks += 1
        if gametest_class not in entrypoint_values(entrypoints.get("fabric-gametest")):
            findings.append(
                finding_type(
                    "BAD_GAMETEST_ENTRYPOINTS",
                    "error",
                    validator._rel(root, metadata_path),
                    f"fabric-gametest must include {gametest_class}.",
                )
            )

    depends = metadata.get("depends")
    checks += 1
    if not isinstance(depends, dict) or depends.get(spec.mod_id) != "*":
        findings.append(
            finding_type(
                "BAD_GAMETEST_DEPENDS",
                "error",
                validator._rel(root, metadata_path),
                f"GameTest metadata must depend on {spec.mod_id!r}.",
            )
        )

    source_path = (
        root
        / "src/gametest/java"
        / Path(*spec.package_name.split("."))
        / f"{class_name(spec.mod_id)}ModGameTests.java"
    )
    checks += 1
    if not source_path.is_file() or source_path.is_symlink():
        findings.append(
            finding_type(
                "MISSING_GAMETEST_SOURCE",
                "error",
                validator._rel(root, source_path),
                "Dedicated Loom GameTest source is missing.",
            )
        )
    return checks
