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
        for behavior in behaviors:
            owner = str(behavior.get("behavior_id") or "")
            if not owner:
                raise ValueError("HOST_SOURCE_BEHAVIOR_ID_REQUIRED")
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
        inspected = callback_interfaces(
            context.get("symbols") or {}, side=side)
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
