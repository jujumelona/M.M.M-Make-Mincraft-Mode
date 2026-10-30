"""Evidence-owned source strategy used when incremental Java cannot proceed.

This is not a fallback prompt or a keyword-specific game template. Every
original runtime read becomes an independent observable producer obligation;
every write becomes a separate native mutation obligation; each behavior
keeps its exact original trigger and effects. A host-generated source unit
must fulfill these obligations and pass compilation and gameplay validation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def prepare_source_owned_strategy(
    context: Mapping[str, Any],
    behaviors: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> dict[str, Any]:
    from .feature_source_work_graph import build_source_work_graph
    from .host_fabric_callback_abi import callback_interfaces

    if isinstance(context.get("source_owned_work_graph"), dict):
        graph = context["source_owned_work_graph"]
    else:
        obligations = []
        inspected = callback_interfaces(
            context.get("symbols") or {}, side=side)
        for behavior in behaviors:
            owner = str(behavior.get("behavior_id") or "")
            if not owner:
                raise ValueError("HOST_SOURCE_BEHAVIOR_ID_REQUIRED")
            # Without a single inspected callback, original occurrence
            # production must precede its listener. An empty ABI catalogue
            # cannot be solved by returning 'register' to the small AI.
            if not inspected and not behavior.get("event_binding"):
                obligations.append({
                    "behavior_id": owner,
                    "reason": "no_inspected_callback",
                    "required_action": (
                        "Materialize an original occurrence source owner "
                        "or an explicitly requested typed player action; "
                        "do not substitute a fabricated Fabric callback."
                    ),
                })
            for name in dict.fromkeys(behavior.get("runtime_state_reads") or ()):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("HOST_SOURCE_OBSERVATION_NAME_INVALID")
                obligations.append({
                    "behavior_id": owner,
                    "name": name,
                    "reason": "no_inspected_semantic_source",
                    "required_action": (
                        "Implement a real typed observation of the named "
                        "runtime state on the inspected occurrence receiver; "
                        "an unbound string or local copy is not a producer."
                    ),
                })
            for name in dict.fromkeys(behavior.get("runtime_state_writes") or ()):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("HOST_SOURCE_MUTATION_NAME_INVALID")
                obligations.append({
                    "behavior_id": owner,
                    "name": name,
                    "reason": "runtime_mutation_requires_verified_effect",
                    "required_action": (
                        "Apply the original state change through a real "
                        "inspected Minecraft mutator after trigger guards; "
                        "the corresponding exact native invocation must be proved."
                    ),
                })
        graph = build_source_work_graph(
            behaviors,
            producer_obligations=obligations,
            inspected_hooks=inspected,
        )
    return {
        **context,
        "host_direct_feature_source": False,
        "host_source_owned": True,
        "source_owned_work_graph": graph,
    }


def prefer_source_owned_strategy(
    router: Any,
    context: Mapping[str, Any],
    behaviors: Sequence[Mapping[str, Any]],
) -> bool:
    """Select the constructive host graph *before* a fragile AI loop.

    Only live constrained llama.cpp coders default to this route; mock and
    non-local transport fixtures keep their historical Java-AST protocol.
    This depends on the actual original state contract, not any game's
    vocabulary or a prompt-specific list of mechanics. The caller can
    explicitly override the strategy for reproducible integration tests.
    """
    preference = context.get("prefer_host_source_owned")
    if preference is not None:
        if type(preference) is not bool:
            raise ValueError("HOST_SOURCE_PREFERENCE_INVALID")
        return preference
    from .fixed_template_generation import _adapter_name
    return (
        _adapter_name(router, "coder") == "llama_cpp"
        and any(
            behavior.get("runtime_state_reads")
            or behavior.get("runtime_state_writes")
            for behavior in behaviors
        )
    )
