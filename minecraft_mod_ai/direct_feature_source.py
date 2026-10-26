"""Member-authoring policy for new gameplay source, without existence gates."""
from __future__ import annotations

import hashlib


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


def behavior_registrar_name(behavior_id):
    return "registerBehavior" + hashlib.sha256(str(behavior_id).strip().encode("utf-8")).hexdigest()[:20]


_MEMBER_PROTOCOL = (
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

_FIELD_VALUE = (
    "Initialize only this field's value with one Java expression matching "
    "declaration_metadata.type and current_unit_purpose. The feature goal "
    "is background context; lifecycle and behavior registration belong to "
    "the declared methods, not this initializer. Do not emit statements, "
    "a field declaration, a callback class, or the surrounding file. "
    "For example, a Set<String> storage field can receive "
    "new java.util.HashSet<>() as its implementation string. An initializer "
    "may construct an object, lambda or anonymous implementation when the "
    "declared type and purpose require one. Supply only the expression "
    "between immutable_signature.prefix and immutable_signature.suffix. "
    "An empty string is allowed only when the host explicitly permits an "
    "uninitialized field. Use declared_member_api for exact helper signatures."
)


def member_implementation_instruction(direct_feature, kind=None):
    if kind == "field":
        return _MEMBER_PROTOCOL + _FIELD_VALUE
    return _body_implementation_instruction(direct_feature)


def _body_implementation_instruction(direct_feature):
    common = _MEMBER_PROTOCOL
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


def plan_direct_support(task, plan_schema, behaviors):
    """Expose already-owned behavior declarations before planning extra members."""
    if not task["host_java_authoring"]:
        return {}
    registrars = {}
    for behavior in behaviors:
        name = behavior_registrar_name(behavior["behavior_id"])
        registrars[name] = direct_behavior_registrar(behavior, name)
    task["host_owned_members"] = [
        {"kind": "method", "name": row["name"], "purpose": row["purpose"],
         "signature": "private void " + row["name"] + "()"}
        for row in registrars.values()
    ]
    properties = plan_schema["properties"]["units"]["items"]["properties"]
    plan_schema["properties"]["units"]["items"]["properties"] = {
        key: properties[key] for key in ("purpose", "kind", "name")}
    return registrars


def support_plan_instruction(direct_feature):
    lifecycle = "authors the required lifecycle body. "
    if not direct_feature:
        return lifecycle + "The host assigns internal helper names. "
    return lifecycle + (
        "host_owned_members already declares every behavior registration method; "
        "their implementations will be authored after the supporting members. "
        "Only plan additional storage, helper logic or nested data types actually "
        "needed by those methods. Zero supporting members is valid. A field "
        "stores a typed value; it cannot stand for the callback class, the "
        "lifecycle method or a behavior registration method. First state its "
        "specific supporting purpose, then choose its declaration kind and name. "
    )
