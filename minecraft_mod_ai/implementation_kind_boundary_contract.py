from __future__ import annotations

"""Keep implementation routing subordinate to evidence and target identity.

``ProductionModule.kind`` selects a generator; it is never semantic authority. Evidence-
owned modules therefore use the bounded custom generator unless a concrete integration
route owns them. Coder localization and mutation-target identity are owned directly by
``progress_aware_tool_loop`` rather than installed here as late monkey-patches.
"""

from collections.abc import Mapping
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
    # custom integration lane. Do not rewrite their identity here.
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



def install(
    *,
    complete_spec_module: Any,
    support_module: Any,
    orchestrator_module: Any,
    template_module: Any,
) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    # One canonical classifier catalog. The planner template may expose the same
    # closed execution choices but must not maintain an independently drifting copy.
    template_module.MODULE_KINDS = complete_spec_module.MODULE_KINDS
    template_module.ASSET_KINDS = complete_spec_module.ASSET_KINDS
    complete_spec_module.IMPLEMENTATION_KINDS = complete_spec_module.MODULE_KINDS

    current = support_module._normalize_modules
    if not getattr(current, "_mmm_semantic_authority_guard", False):

        @wraps(current)
        def normalize_modules(modules: tuple[Any, ...], spec: Any):
            routed = tuple(
                _route_evidence_owned_custom(
                    module, complete_spec_module.ProductionModule
                )
                for module in modules
            )
            ordered, receipts = current(routed, spec)
            routed_ids = {
                module.module_id: str(module.config.get("requested_kind") or "")
                for module in ordered
                if module.kind == "custom_java"
                and str(module.config.get("semantic_authority") or "")
                == "evidence_task"
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

        # complete_orchestrator imported the function by name. Replace only the exact
        # old alias so unrelated wrappers remain intact.
        if getattr(orchestrator_module, "_normalize_modules", None) is current:
            orchestrator_module._normalize_modules = normalize_modules

    _INSTALLED = True


__all__ = [
    "_is_evidence_owned",
    "_route_evidence_owned_custom",
    "install",
]
