from __future__ import annotations

"""Prevent implementation classifier names from becoming semantic authority.

``ProductionModule.kind`` selects a generator.  Names such as ``quest`` or ``gui`` are
therefore implementation classifications, not permission to replace an evidence-owned
task with the semantics of a built-in template.  Evidence-backed modules are routed to
the bounded custom generator unless an explicit non-semantic integration route owns the
implementation.  Legacy proposals without an evidence task retain their old built-in
routes for compatibility.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_INSTALLED = False


def _is_evidence_owned(module: Any) -> bool:
    config = getattr(module, "config", None)
    if not isinstance(config, Mapping):
        return False
    task = config.get("evidence_task")
    if isinstance(task, Mapping) and task:
        return True
    return bool(
        str(config.get("evidence_plan_sha256") or "").strip()
        and config.get("requirement_refs")
    )


def _route_evidence_owned_custom(module: Any, production_module_type: Any) -> Any:
    if not _is_evidence_owned(module):
        return module
    kind = str(getattr(module, "kind", "") or "")
    config = dict(getattr(module, "config", {}) or {})
    # Integrations already require a concrete integration_type or go through the
    # custom integration lane.  Do not rewrite their identity here.
    if kind in {"custom_java", "integration"}:
        return module
    config.setdefault("requested_kind", kind)
    config["implementation_classifier_role"] = "routing_hint_only"
    config["semantic_authority"] = "evidence_task"
    return production_module_type(
        module_id=module.module_id,
        kind="custom_java",
        config=config,
        depends_on=module.depends_on,
        required_gates=module.required_gates,
    )


def _fresh_owned_symbol_context(payload: Any, tool_loop: Any) -> Any | None:
    """Project a fresh evidence task's host-reserved source anchor into coder state.

    Fresh work is creation, not mutation of an already-existing file.  The evidence
    planner already reserved the exact source locator, so repository search must never
    be used to guess another file (for example ``fabric.mod.json``) or to prove that a
    not-yet-created Java file exists.  Adapt/reuse work deliberately does not enter this
    path and retains the normal file -> symbol -> body localization contract.
    """

    if not isinstance(payload, Mapping):
        return None
    module = payload.get("module")
    if not isinstance(module, Mapping):
        return None
    config = module.get("config")
    if not isinstance(config, Mapping):
        return None
    task = config.get("evidence_task")
    if not isinstance(task, Mapping):
        return None
    bindings = task.get("production_bindings")
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes, bytearray)):
        return None

    production_bindings = [item for item in bindings if isinstance(item, Mapping)]
    if not production_bindings:
        return None
    actions = {
        str(item.get("reuse_action") or "").strip().casefold()
        for item in production_bindings
        if str(item.get("reuse_action") or "").strip()
    }
    # Mixed/unknown action ownership is not safe to reinterpret as file creation.
    if actions != {"fresh"}:
        return None

    for binding in production_bindings:
        anchors = binding.get("owned_anchors")
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes, bytearray)):
            continue
        for anchor in anchors:
            if not isinstance(anchor, Mapping) or str(anchor.get("kind") or "") != "symbol":
                continue
            locator = str(anchor.get("locator") or "").strip().replace("\\", "/")
            target_path, separator, target_symbol = locator.partition("#")
            target_path = target_path.strip()
            if (
                not target_path
                or target_path.startswith("/")
                or ".." in target_path.split("/")
                or not tool_loop._is_workspace_file_path(target_path)
            ):
                continue
            return tool_loop.TargetMutationContext(
                target_path=target_path,
                target_symbol=target_symbol.strip() if separator and target_symbol.strip() else None,
                is_new_file=True,
                evidence_source="evidence_fresh_owned_anchor",
            )
    return None


def _install_fresh_owned_target_grounding() -> None:
    """Make host-reserved fresh targets outrank incidental RAG/source observations."""

    from . import progress_aware_tool_loop as tool_loop

    current = tool_loop._extract_mutation_context_from_payload
    if getattr(current, "_mmm_fresh_owned_target_grounding", False):
        return

    @wraps(current)
    def extract_mutation_context(payload: Any):
        fresh = _fresh_owned_symbol_context(payload, tool_loop)
        if fresh is not None:
            return fresh
        return current(payload)

    extract_mutation_context._mmm_fresh_owned_target_grounding = True
    extract_mutation_context.__wrapped__ = current
    tool_loop._extract_mutation_context_from_payload = extract_mutation_context


def install(*, complete_spec_module: Any, support_module: Any, orchestrator_module: Any, template_module: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    # One canonical classifier catalog.  The planner template may expose the same
    # closed execution choices but must not maintain an independently drifting copy.
    template_module.MODULE_KINDS = complete_spec_module.MODULE_KINDS
    template_module.ASSET_KINDS = complete_spec_module.ASSET_KINDS
    complete_spec_module.IMPLEMENTATION_KINDS = complete_spec_module.MODULE_KINDS

    current = support_module._normalize_modules
    if not getattr(current, "_mmm_semantic_authority_guard", False):
        @wraps(current)
        def normalize_modules(modules: tuple[Any, ...], spec: Any):
            routed = tuple(
                _route_evidence_owned_custom(module, complete_spec_module.ProductionModule)
                for module in modules
            )
            ordered, receipts = current(routed, spec)
            routed_ids = {
                module.module_id: str(module.config.get("requested_kind") or "")
                for module in ordered
                if module.kind == "custom_java"
                and str(module.config.get("semantic_authority") or "") == "evidence_task"
            }
            if routed_ids:
                receipts = list(receipts)
                receipts.extend(
                    {
                        "schema_version": "mmm/implementation-kind-boundary-v1",
                        "status": "ROUTED_CUSTOM",
                        "module_id": module_id,
                        "requested_kind": requested_kind,
                        "classifier_role": "routing_hint_only",
                        "semantic_authority": "evidence_task",
                    }
                    for module_id, requested_kind in sorted(routed_ids.items())
                )
            return ordered, receipts

        normalize_modules._mmm_semantic_authority_guard = True
        normalize_modules.__wrapped__ = current
        support_module._normalize_modules = normalize_modules

        # complete_orchestrator imported the function by name.  Replace only the exact
        # old alias so unrelated wrappers remain intact.
        if getattr(orchestrator_module, "_normalize_modules", None) is current:
            orchestrator_module._normalize_modules = normalize_modules

    _install_fresh_owned_target_grounding()
    _INSTALLED = True


__all__ = [
    "_fresh_owned_symbol_context",
    "_is_evidence_owned",
    "_route_evidence_owned_custom",
    "install",
]
