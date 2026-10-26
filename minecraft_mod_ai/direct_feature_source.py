"""Member-authoring policy for new gameplay source, without existence gates."""
from __future__ import annotations


def direct_feature_mode(spec, context):
    return bool(context is not None and context.get("host_direct_feature_source")
                and spec.get("bindings", {}).get("request_feature_entrypoint"))


def direct_behavior_registrar(behavior, name):
    return {
        "kind": "method", "name": name,
        "purpose": (
            "Implement and register this behavior's actual activation: "
            + str(behavior.get("trigger") or behavior["intent"])
            + ". Implement its effects: " + str(behavior.get("effects") or behavior["intent"])
            + ". Connect the declared helpers and actual target APIs; create the "
            "required input/observer logic instead of searching for an already "
            "implemented gameplay method."
        ),
        "host_required_behavior": behavior["behavior_id"],
    }


def member_implementation_instruction(direct_feature):
    common = (
        "Return JSON with one implementation string containing only the field "
        "initializer or method/constructor body. The host fixes the signature "
        "and adds all declaration delimiters. Do not repeat the signature. "
        "An empty string may select an approved uninitialized field. "
        "Never return package/import directives, an outer class or extra members. "
        "Respect implementation_requirements, side, current_unit_purpose and "
        "actual prerequisite/member APIs. For shared state use only host_state_access "
        "typed readState_/writeState_ methods; the host owns state keys, value casts "
        "and context signatures. "
    )
    if direct_feature:
        common += (
            "New feature-local state and helpers are the actual declarations in "
            "declared_member_api. Implement them directly. Register real event "
            "listeners or interactions and connect every effect to its activation. "
            "An unavailable custom gameplay method is work to implement, not a "
            "reason to decline or return SOURCE_REQUIRED. Use retained UI action "
            "contracts when the feature already has an authored input. "
        )
    else:
        common += (
            "An undeclared work-order state is NOT a Java accessor: use only "
            "verified callback/world inputs when that preserves behavior, never "
            "substitute an unrelated state field. Persistent state absent from "
            "the canonical plan must be resolved upstream, not fabricated in Java. "
        )
    return common + (
        "declared_member_api contains the exact signatures of all helpers, "
        "including forward references. Implement the requested behavior "
        "without placeholders or TODOs."
    )
