from __future__ import annotations

"""Model-free production backend for authored Typed PlanIR modules."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .complete_spec import ProductionModule
from .project_edit import inspect_fabric_project, write_text_files
from .typed_plan_java import render_typed_plan_java


def _typed_program_path(package_name: str) -> str:
    return (
        "src/main/java/"
        + str(package_name).replace(".", "/")
        + "/AuthoredProgram.java"
    )


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
    patch_receipt = write_text_files(
        info,
        {expected_path: source},
        replace_existing=True,
    )
    return {
        "schema_version": "mmm/custom-module-result-v3",
        "module_id": module.module_id,
        "kind": module.kind,
        "status": "SOURCE_GENERATED",
        "touched_paths": [expected_path],
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
