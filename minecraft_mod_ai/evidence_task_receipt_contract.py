from __future__ import annotations

"""Canonical receipt extensions shared by evidence-first production and reuse validation.

The evidence-first handoff owns the four host-only receipt extensions attached to a
semantic task before it becomes a production module.  Older reuse validation predates
those extensions and only understands the semantic task plus ``request_context``.
This contract validates the complete receipt once, then presents the legacy validator
with a compatibility view without mutating the approved proposal.
"""

from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

from .evidence_first_handoff import (
    build_evidence_first_handoff,
    validate_evidence_first_handoff,
)
from .evidence_first_planning import EvidencePlanError, validate_evidence_first_plan

RECEIPT_EXTENSION_FIELDS = frozenset(
    {
        "handoff_sha256",
        "production_bindings",
        "asset_bindings",
        "request_context",
    }
)

_INSTALLED = False
_ACTIVE_RECEIPTS: ContextVar[
    tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]] | None
] = ContextVar(
    "mmm_evidence_task_receipts",
    default=None,
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def build_task_receipt_extensions(
    plan: Mapping[str, Any],
    *,
    handoff: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the exact handoff-owned receipt extensions for every semantic task."""

    validate_evidence_first_plan(plan)
    resolved_handoff = (
        dict(handoff) if handoff is not None else build_evidence_first_handoff(plan)
    )
    validate_evidence_first_handoff(resolved_handoff, source_plan=plan)

    request_catalog = _mapping(plan.get("request_catalog"))
    raw_requirements = request_catalog.get("requirements")
    raw_tasks = plan.get("tasks")
    if not isinstance(raw_requirements, list) or not isinstance(raw_tasks, list):
        raise EvidencePlanError(
            "Evidence task receipts require list requirement and task catalogs."
        )

    requirements = {
        str(item.get("requirement_id") or ""): dict(item)
        for item in raw_requirements
        if isinstance(item, Mapping) and str(item.get("requirement_id") or "")
    }
    if len(requirements) != len(raw_requirements):
        raise EvidencePlanError(
            "Evidence task receipt requirements contain invalid or duplicate references."
        )

    tasks = {
        str(item.get("task_id") or ""): dict(item)
        for item in raw_tasks
        if isinstance(item, Mapping) and str(item.get("task_id") or "")
    }
    if len(tasks) != len(raw_tasks):
        raise EvidencePlanError(
            "Evidence task receipt catalog contains invalid or duplicate task references."
        )

    graph = _mapping(resolved_handoff.get("work_graph"))
    task_refs = _strings(graph.get("task_refs"))
    if tuple(tasks) != task_refs:
        raise EvidencePlanError(
            "Evidence task receipt order drifted from the validated handoff graph."
        )

    production_by_task: dict[str, list[dict[str, Any]]] = {
        task_ref: [] for task_ref in task_refs
    }
    for item in resolved_handoff.get("production_modules", ()):
        if not isinstance(item, Mapping):
            raise EvidencePlanError(
                "Evidence production receipt binding must be an object."
            )
        task_ref = str(item.get("task_ref") or "")
        if task_ref not in production_by_task:
            raise EvidencePlanError(
                f"Evidence production receipt references unknown task {task_ref!r}."
            )
        production_by_task[task_ref].append(dict(item))

    assets_by_task: dict[str, list[dict[str, Any]]] = {
        task_ref: [] for task_ref in task_refs
    }
    for item in resolved_handoff.get("asset_requests", ()):
        if not isinstance(item, Mapping):
            raise EvidencePlanError("Evidence asset receipt binding must be an object.")
        task_ref = str(item.get("task_ref") or "")
        if task_ref not in assets_by_task:
            raise EvidencePlanError(
                f"Evidence asset receipt references unknown task {task_ref!r}."
            )
        assets_by_task[task_ref].append(dict(item))

    handoff_sha256 = str(resolved_handoff.get("handoff_sha256") or "")
    if not handoff_sha256:
        raise EvidencePlanError("Evidence task receipt is missing the handoff hash.")

    result: dict[str, dict[str, Any]] = {}
    for task_ref in task_refs:
        task = tasks[task_ref]
        requirement_refs = _strings(task.get("requirement_refs"))
        unknown = [
            reference for reference in requirement_refs if reference not in requirements
        ]
        if unknown:
            raise EvidencePlanError(
                f"Evidence task {task_ref!r} receipt references unknown requirements {unknown!r}."
            )
        result[task_ref] = {
            "handoff_sha256": handoff_sha256,
            "production_bindings": production_by_task[task_ref],
            "asset_bindings": assets_by_task[task_ref],
            "request_context": {
                "prompt_sha256": request_catalog.get("prompt_sha256"),
                "requirements": [
                    requirements[reference] for reference in requirement_refs
                ],
            },
        }
    return result


def validate_task_receipt(
    embedded: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    expected_extensions: Mapping[str, Any],
) -> None:
    """Fail closed unless an embedded task is the exact semantic task plus extensions."""

    task_id = str(task.get("task_id") or "")
    extras = set(embedded) - set(task)
    unknown = extras - RECEIPT_EXTENSION_FIELDS
    if unknown:
        raise EvidencePlanError(
            f"Evidence task {task_id!r} contains unrecognized receipt fields: {sorted(unknown)}."
        )
    missing = RECEIPT_EXTENSION_FIELDS - set(embedded)
    if missing:
        raise EvidencePlanError(
            f"Evidence task {task_id!r} is missing receipt fields: {sorted(missing)}."
        )
    for key, value in task.items():
        if embedded.get(key) != value:
            raise EvidencePlanError(
                f"Evidence task {task_id!r} changed host-owned field {key!r}."
            )
    for key in RECEIPT_EXTENSION_FIELDS:
        if embedded.get(key) != expected_extensions.get(key):
            raise EvidencePlanError(
                f"Evidence task {task_id!r} has stale or mismatched receipt field {key!r}."
            )


def _install_reuse_receipt_guard() -> None:
    from . import resource_asset_production as production

    current_bind = production._bind_evidence_reuse_plan
    if not getattr(current_bind, "_mmm_evidence_task_receipt_context", False):

        @wraps(current_bind)
        def bind_evidence_reuse_plan(proposal: Any, evidence_plan: Mapping[str, Any]):
            expected = build_task_receipt_extensions(evidence_plan)
            token = _ACTIVE_RECEIPTS.set((evidence_plan, expected))
            try:
                return current_bind(proposal, evidence_plan)
            finally:
                _ACTIVE_RECEIPTS.reset(token)

        bind_evidence_reuse_plan._mmm_evidence_task_receipt_context = True  # type: ignore[attr-defined]
        production._bind_evidence_reuse_plan = bind_evidence_reuse_plan

    current_validate = production._validate_evidence_module_binding
    if getattr(current_validate, "_mmm_evidence_task_receipt_guard", False):
        return

    @wraps(current_validate)
    def validate_evidence_module_binding(*args: Any, **kwargs: Any) -> None:
        context = _ACTIVE_RECEIPTS.get()
        if context is None:
            return current_validate(*args, **kwargs)
        _plan, expected = context

        module = kwargs.get("module")
        task = kwargs.get("task")
        if module is None and args:
            module = args[0]
        if task is None and len(args) > 1:
            task = args[1]
        if module is None or not isinstance(task, Mapping):
            return current_validate(*args, **kwargs)

        task_id = str(task.get("task_id") or "")
        if task_id not in expected:
            raise EvidencePlanError(
                f"Evidence reuse validation references unknown task {task_id!r}."
            )

        config = getattr(module, "config", None)
        embedded = config.get("evidence_task") if isinstance(config, Mapping) else None
        if not isinstance(embedded, Mapping):
            raise EvidencePlanError(
                f"Evidence task {task_id!r} has no canonical host-owned receipt."
            )
        validate_task_receipt(
            embedded,
            task=task,
            expected_extensions=expected[task_id],
        )

        compatibility_receipt = dict(task)
        compatibility_receipt["request_context"] = expected[task_id]["request_context"]
        compatibility_module = replace(
            module,
            config={**dict(config), "evidence_task": compatibility_receipt},
        )

        if "module" in kwargs:
            kwargs = {**kwargs, "module": compatibility_module}
            return current_validate(*args, **kwargs)
        if args:
            mutable_args = list(args)
            mutable_args[0] = compatibility_module
            return current_validate(*mutable_args, **kwargs)
        return current_validate(module=compatibility_module, **kwargs)

    validate_evidence_module_binding._mmm_evidence_task_receipt_guard = True  # type: ignore[attr-defined]
    production._validate_evidence_module_binding = validate_evidence_module_binding


def install() -> None:
    """Install the shared task-receipt contract exactly once on the live path."""

    global _INSTALLED
    if _INSTALLED:
        return
    _install_reuse_receipt_guard()
    _INSTALLED = True


__all__ = [
    "RECEIPT_EXTENSION_FIELDS",
    "build_task_receipt_extensions",
    "install",
    "validate_task_receipt",
]
