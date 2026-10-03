from __future__ import annotations

"""Model-free production backend for authored Typed PlanIR modules."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .complete_spec import ProductionModule
from .production_state_compiler import render_production_state_java
from .project_edit import inspect_fabric_project, write_text_files
from .typed_plan_ir import typed_plan_uses_state
from .typed_plan_java import render_typed_plan_java


def _typed_program_path(package_name: str) -> str:
    return (
        "src/main/java/"
        + str(package_name).replace(".", "/")
        + "/AuthoredProgram.java"
    )


def _assert_host_owned_or_absent(
    root: Path,
    relative: str,
    *,
    marker: str,
) -> None:
    target = root / relative
    if not target.exists():
        return
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"TYPED_PLAN_TARGET_INVALID: {relative}")
    current = target.read_text(encoding="utf-8")
    if marker not in current:
        raise ValueError(f"TYPED_PLAN_OWNERSHIP_CONFLICT: {relative}")


def generate_typed_plan_module(
    project_root: str | Path,
    *,
    module: ProductionModule,
) -> dict[str, Any]:
    """Compile a persisted Typed PlanIR directly into Java with no model access."""

    module.validate()
    config = module.config
    raw_plan = config.get("typed_plan_ir")
    if not isinstance(raw_plan, Mapping) or not raw_plan:
        raise ValueError("TYPED_PLAN_IR_REQUIRED")

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("TYPED_PLAN_PROJECT_REQUIRED")
    info = inspect_fabric_project(root)

    package_name = str(
        config.get("typed_plan_package") or info.package_name
    ).strip()
    if package_name != info.package_name:
        raise ValueError(
            "TYPED_PLAN_PACKAGE_MISMATCH: "
            f"{package_name!r} != {info.package_name!r}"
        )
    expected_path = _typed_program_path(package_name)
    configured_path = str(
        config.get("typed_plan_path") or expected_path
    ).replace("\\", "/").strip()
    if configured_path != expected_path:
        raise ValueError(
            "TYPED_PLAN_PATH_MISMATCH: "
            f"{configured_path!r} != {expected_path!r}"
        )

    target = root / expected_path
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError("TYPED_PLAN_TARGET_INVALID")
        current = target.read_text(encoding="utf-8")
        if "// MMM:TYPED_PLAN_OWNER" not in current:
            raise ValueError("TYPED_PLAN_OWNERSHIP_CONFLICT")

    raw_capabilities = config.get("typed_plan_capabilities")
    capabilities = (
        dict(raw_capabilities)
        if isinstance(raw_capabilities, Mapping)
        else {}
    )
    source = render_typed_plan_java(
        raw_plan,
        package=package_name,
        capabilities=capabilities,
    )
    files = {expected_path: source}
    _assert_host_owned_or_absent(
        root,
        expected_path,
        marker="// MMM:TYPED_PLAN_OWNER",
    )

    if typed_plan_uses_state(raw_plan):
        raw_state = config.get("typed_plan_state_section")
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise ValueError("TYPED_PLAN_STATE_AUTHORITY_REQUIRED")
        state_path = (
            "src/main/java/"
            + package_name.replace(".", "/")
            + "/AuthoredStateModel.java"
        )
        _assert_host_owned_or_absent(
            root,
            state_path,
            marker="// MMM:TYPED_PLAN_STATE_OWNER",
        )
        files[state_path] = render_production_state_java(
            raw_state,
            package_name=package_name,
        )

    patch_receipt = write_text_files(
        info,
        files,
        replace_existing=True,
    )
    touched_paths = sorted(files)
    return {
        "schema_version": "mmm/custom-module-result-v3",
        "module_id": module.module_id,
        "kind": module.kind,
        "status": "SOURCE_GENERATED",
        "touched_paths": touched_paths,
        "patch_receipt": patch_receipt,
        "operation_count": len(patch_receipt.get("operations") or ()),
        "required_gates": list(module.required_gates),
        "generation_verification": {
            "status": "PASS",
            "mode": "host_typed_plan_ir_deferred_to_pipeline",
            "compile_deferred": True,
            "model_calls": 0,
        },
        "runtime_tests": [
            "Execute the requested GameTest/runtime gates."
        ],
    }


__all__ = ["generate_typed_plan_module"]
