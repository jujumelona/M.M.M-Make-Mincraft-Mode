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


def _materialized_read_bindings(context: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Reuse only canonical state accessors proven by actual declarations."""
    import re
    scope = context.get("host_state_work_order_scope") or {}
    if not isinstance(scope, Mapping):
        return {}
    allowed = scope.get("declared_readable") or ()
    if not isinstance(allowed, (list, tuple, set)):
        return {}
    allowed = {name for name in allowed if isinstance(name, str)}
    result: dict[str, dict[str, str]] = {}
    for row in context.get("host_state_access") or ():
        if not isinstance(row, Mapping):
            continue
        owner = row.get("owner")
        if not isinstance(owner, str) or not owner:
            continue
        for method in row.get("methods") or ():
            if not isinstance(method, Mapping):
                continue
            symbol = method.get("name")
            if not isinstance(symbol, str) or not symbol.startswith("readState_"):
                continue
            slot = symbol[len("readState_"):]
            if (slot not in allowed
                    or not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", slot)
                    or not isinstance(method.get("signature"), str)
                    or not re.search(r"\bpublic\s+static\b", method["signature"])
                    or not re.search(r"\b" + re.escape(symbol)
                                     + r"\s*\(\s*\)", method["signature"])):
                continue
            result[slot] = {"owner": owner, "method": symbol,
                            "signature": method["signature"]}
    return result


def prepare_source_owned_strategy(
    context: Mapping[str, Any],
    behaviors: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> dict[str, Any]:
    from .feature_source_work_graph import build_source_work_graph
    from .host_fabric_callback_abi import callback_interfaces
    from .source_owned_hook import _event_candidates_for_occurrence

    if isinstance(context.get("source_owned_work_graph"), dict):
        graph = context["source_owned_work_graph"]
    else:
        obligations = []
        existing_reads = _materialized_read_bindings(context)
        reused_reads = {}
        inspected = callback_interfaces(
            context.get("symbols") or {}, side=side)
        for behavior in behaviors:
            owner = str(behavior.get("behavior_id") or "")
            if not owner:
                raise ValueError("HOST_SOURCE_BEHAVIOR_ID_REQUIRED")
            # Without a single inspected callback, original occurrence
            # production must precede its listener. An empty ABI catalogue
            # cannot be solved by returning 'register' to the small AI.
            matching_occurrence = _event_candidates_for_occurrence(behavior, inspected)
            if not matching_occurrence and not behavior.get("event_binding"):
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
                declared_reads = behavior.get("state_reads") or ()
                if (isinstance(declared_reads, (list, tuple, set))
                        and name in existing_reads
                        and name in declared_reads):
                    reused_reads.setdefault(owner, {})[name] = existing_reads[name]
                    continue
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
    result = {
        **context,
        "host_direct_feature_source": False,
        "host_source_owned": True,
        "source_owned_work_graph": graph,
    }
    if not isinstance(context.get("source_owned_work_graph"), dict):
        result["source_owned_reused_reads"] = reused_reads
    return result


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
    # All live 9B gameplay features need host-owned occurrence, lifecycle,
    # data and effect decomposition. A feature without an explicit world-state
    # annotation can still require a real callback and executable effect;
    # routing it back to the 96-choice AST planner recreates the original bug.
    return _adapter_name(router, "coder") == "llama_cpp" and bool(behaviors)
