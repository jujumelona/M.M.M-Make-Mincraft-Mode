from __future__ import annotations

"""Keep implementation routing subordinate to evidence and target identity.

``ProductionModule.kind`` selects a generator; it is never semantic authority. Evidence-
owned modules therefore use the bounded custom generator unless a concrete integration
route owns them. The late runtime contract also hardens coder target state so a planned
fresh file cannot bypass repository reuse evidence and observations from different files
cannot be merged into one mutation context.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

_INSTALLED = False
_EXISTING_TARGET_EVIDENCE_PREFIXES = (
    "search_code_rag",
    "sources_",
    "java_workspace_symbols",
    "observation_page_",
    "files_",
)


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


def _fresh_owned_symbol_context(payload: Any, tool_loop: Any) -> Any | None:
    """Compatibility delegate to the single host-owned fresh-target resolver."""

    return tool_loop._fresh_owned_symbol_context(payload)


def _sequence_has_values(value: Any) -> bool:
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and value
    )


def _fresh_target_has_reuse_evidence(payload: Any) -> bool:
    """Detect evidence that contradicts treating an evidence task as a new file.

    The evidence planner may reserve a fresh anchor, but an execution projection that
    also carries component/source reuse references is internally inconsistent. Fail
    closed to repository localization instead of letting ``is_new_file`` skip source
    body grounding.
    """

    if not isinstance(payload, Mapping):
        return False
    module = payload.get("module")
    if not isinstance(module, Mapping):
        return False
    config = module.get("config")
    if not isinstance(config, Mapping):
        return False
    task = config.get("evidence_task")
    if not isinstance(task, Mapping):
        return False

    for key in ("reuse_refs", "component_refs", "source_refs"):
        if _sequence_has_values(task.get(key)):
            return True

    bindings = task.get("production_bindings")
    if not isinstance(bindings, Sequence) or isinstance(
        bindings, (str, bytes, bytearray)
    ):
        return False
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        if str(binding.get("reuse_action") or "").strip().casefold() != "fresh":
            continue
        for key in ("reuse_refs", "component_refs", "source_refs"):
            if _sequence_has_values(binding.get(key)):
                return True
    return False


def _normalized_target_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _proves_existing_target(context: Any) -> bool:
    if bool(getattr(context, "is_new_file", False)):
        return False
    if not _normalized_target_path(getattr(context, "target_path", None)):
        return False
    if getattr(context, "source_body", None):
        return True
    source = str(getattr(context, "evidence_source", "") or "")
    return source.startswith(_EXISTING_TARGET_EVIDENCE_PREFIXES)


def _initial_exact_target_context(
    payload: Any,
    *,
    prospective: Any,
    extract: Any,
) -> Any | None:
    """Return already-supplied exact source when it is the reserved target itself.

    ``CustomModuleGenerator`` ranks the cached ``ProjectIndex`` with the full module
    query before the coder turn. Reuse that bounded exact-source page instead of doing
    another repository walk. Incidental files remain irrelevant: only an exact path
    match can overturn the prospective new-file assumption.
    """

    if not isinstance(payload, Mapping):
        return None
    initial = payload.get("initial_exact_source_context")
    if not isinstance(initial, (Mapping, list, tuple)):
        return None
    target_path = _normalized_target_path(prospective.target_path)
    if not target_path:
        return None

    candidates: list[Any] = []
    if isinstance(initial, Mapping):
        for key in ("global_anchors", "records", "excerpts", "files", "sources", "hits", "results"):
            value = initial.get(key)
            if isinstance(value, Mapping):
                # ``files`` may be a path -> source mapping. Keep each entry bounded.
                candidates.extend(
                    {"path": str(path), "content": content}
                    for path, content in value.items()
                )
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                candidates.extend(value)
        candidates.append(initial)
    else:
        candidates.extend(initial)

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_path = _normalized_target_path(
            candidate.get("source_path")
            or candidate.get("path")
            or candidate.get("file")
            or candidate.get("uri")
        )
        candidate_path = candidate_path.removeprefix("file:///").removeprefix("file://")
        if candidate_path != target_path:
            continue
        existing = extract({"hits": [candidate]})
        if existing is None or _normalized_target_path(existing.target_path) != target_path:
            existing = prospective.__class__(
                target_path=target_path,
                target_symbol=prospective.target_symbol,
                is_new_file=False,
                evidence_source="initial_exact_target_path",
            )
        elif prospective.target_symbol and not existing.target_symbol:
            existing = replace(existing, target_symbol=prospective.target_symbol)
        return replace(existing, is_new_file=False)
    return None


def _install_target_context_hardening() -> None:
    """Harden fresh/reuse routing and file-identity-preserving context merges."""

    from . import progress_aware_tool_loop as tool_loop

    current_extract = tool_loop._extract_mutation_context_from_payload
    if not getattr(current_extract, "_mmm_fresh_owned_target_grounding", False):

        @wraps(current_extract)
        def extract_mutation_context(payload: Any):
            context = current_extract(payload)
            if context is None or not context.is_new_file:
                return context
            if _fresh_target_has_reuse_evidence(payload):
                return tool_loop.TargetMutationContext(
                    evidence_source="reuse_evidence_requires_localization"
                )
            existing = _initial_exact_target_context(
                payload, prospective=context, extract=current_extract
            )
            return existing or context

        # Keep the historical marker: runtime integrity/tests use it to identify the
        # one late fresh-target boundary, while semantics are now fail-closed on reuse.
        extract_mutation_context._mmm_fresh_owned_target_grounding = True
        extract_mutation_context.__wrapped__ = current_extract
        tool_loop._extract_mutation_context_from_payload = extract_mutation_context

    context_type = tool_loop.TargetMutationContext
    current_merge = context_type.merge
    if getattr(current_merge, "_mmm_target_identity_merge_guard", False):
        return

    @wraps(current_merge)
    def merge_target_context(self: Any, other: Any):
        self_path = _normalized_target_path(self.target_path)
        other_path = _normalized_target_path(other.target_path)

        # A newly localized different file starts a new localization chain. Carrying
        # the previous file's symbol/body/line range into it can authorize an edit
        # against source that belongs to another target.
        if self_path and other_path and self_path != other_path:
            return other

        merged = current_merge(self, other)
        if other.is_new_file:
            return merged
        if _proves_existing_target(other) and merged.is_new_file:
            # Exact repository evidence for the same path overrides an earlier
            # prospective-new-file assumption. This restores body grounding.
            return replace(merged, is_new_file=False)
        return merged

    merge_target_context._mmm_target_identity_merge_guard = True
    merge_target_context.__wrapped__ = current_merge
    context_type.merge = merge_target_context


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

    _install_target_context_hardening()
    _INSTALLED = True


__all__ = [
    "_fresh_owned_symbol_context",
    "_fresh_target_has_reuse_evidence",
    "_is_evidence_owned",
    "_route_evidence_owned_custom",
    "install",
]
