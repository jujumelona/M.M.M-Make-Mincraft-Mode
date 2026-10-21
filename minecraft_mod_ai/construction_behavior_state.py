"""Resolve behavior state ownership before freezing production source jobs.

State references originate in semantic planning, after the saved state model
was authored. The Java writer must not resolve that disagreement by inventing
an accessor, changing another value, or treating world blocks as a string.
This boundary obtains explicit semantic bindings, validates canonical scalar
declarations and retains event/world obligations separately from host storage.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import replace

from .construction_direct_writer import _digest, _read_checkpoint, _save_checkpoint
from .fixed_template_generation import _validate_native_arguments
from .structured_state_runtime import (
    state_concern_schema,
    state_variable_default_schema,
    validate_structured_state_section,
)
from .typed_host_generation_contract import (
    canonical_typed_state_store_config,
    normalize_typed_state_store_config,
)
from .typed_platform_ir import validate_platform_modules


def _identifier_parts(value):
    """Compare identifier spellings; never infer two gameplay concepts are synonyms."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
    return tuple(re.findall(r"[^\W_]+", spaced.casefold()))


def _canonical_targets(reference, variables):
    parts = _identifier_parts(reference)
    if not parts:
        return ()
    matches = set()
    for row in variables:
        name = _identifier_parts(row["name"])
        owner = _identifier_parts(row.get("owner", ""))
        if parts == name or (owner and parts == owner + name):
            matches.add(row["name"])
    return tuple(sorted(matches))


def _validate_replayed_binding(reference, binding, variables):
    """Revalidate saved LLM choices against the current state authority."""
    if not isinstance(binding, dict) or binding.get("original_name") != reference:
        raise ValueError("BEHAVIOR_STATE_BINDING_SOURCE_MISMATCH")
    kind = binding.get("binding")
    if kind == "canonical_state":
        targets = _canonical_targets(reference, variables)
        if len(targets) != 1 or binding.get("state_name") != targets[0]:
            raise ValueError("BEHAVIOR_STATE_CANONICAL_MEANING_UNPROVEN: " + reference)
    elif kind == "world_state":
        observation = binding.get("observation")
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError("BEHAVIOR_STATE_WORLD_OBSERVATION_REQUIRED: " + reference)
    elif kind == "new_scalar_state":
        row = binding.get("variable")
        if not isinstance(row, dict) or row.get("name") != binding.get("state_name"):
            raise ValueError("BEHAVIOR_STATE_NEW_SCALAR_INVALID: " + reference)
    else:
        raise ValueError("BEHAVIOR_STATE_BINDING_KIND_INVALID: " + reference)


def _choose(router, field, schema, context):
    page = {"type": "object", "properties": {field: schema},
            "required": [field], "additionalProperties": False}
    response = router.generate_structured_data(
        "planner", [{"role": "system", "content": (
            "Resolve ONE behavior state reference before the host compiles Java. "
            "The host resolves existing canonical identifiers deterministically, not "
            "by guessing synonyms. new_scalar_state declares a genuinely missing scalar "
            "owned by the host. world_state keeps event/world/entity/inventory/block/UI "
            "observations and mutations in their existing runtime owner. Do not create "
            "a duplicate scalar for world objects or encode collections as strings. "
            "Preserve every original read/write obligation. Author only the requested "
            "field; no Java source or wrapper. Choices are semantic planning, not "
            "runtime verification.")},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        data_name="resolve_behavior_state_" + field, parameters=page,
    )
    return _validate_native_arguments(response, page)[field]


def _binding(router, name, uses, variables):
    context = {"original_state_reference": name, "required_uses": uses,
               "canonical_variables": variables}
    targets = _canonical_targets(name, variables)
    if len(targets) > 1:
        raise ValueError("BEHAVIOR_STATE_CANONICAL_IDENTIFIER_AMBIGUOUS: " + name)
    if targets:
        # No model-selected aliases between unrelated canonical scalars.
        return {"original_name": name, "binding": "canonical_state",
                "state_name": targets[0],
                "semantic_status": "host_verified_identifier_match"}
    kind = _choose(router, "binding", {
        "type": "string", "enum": ["new_scalar_state", "world_state"]}, context)
    result = {"original_name": name, "binding": kind,
              "semantic_status": "model_authored_unverified"}
    if kind == "world_state":
        result["observation"] = _choose(router, "observation", {
            "type": "string", "minLength": 1, "maxLength": 320}, context)
    else:
        # The reference is semantic data, not necessarily an identifier. Keep
        # it in the binding record while the host fixes one legal stable name.
        target_name = (name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name)
                       else "state_" + hashlib.sha256(name.encode()).hexdigest()[:20])
        if target_name in {row["name"] for row in variables}:
            raise ValueError("BEHAVIOR_STATE_IDENTIFIER_COLLISION: " + name)
        row = {"name": target_name}
        schema = state_concern_schema("variables")["properties"]
        for field in ("owner", "type", "unit", "default", "domain"):
            field_schema = (state_variable_default_schema(row["type"]) if field == "default"
                            else schema[field])
            row[field] = _choose(router, field, field_schema, {**context, "fixed_variable": row})
        validate_structured_state_section({"variables": [*variables, row]})
        result.update(state_name=target_name, variable=row)
    return result


def _rebind_state_stores(plan, sections):
    """Validate the old storage contract before deriving one for new authority.

    Schema versions and migration destinations are host-owned projections of
    canonical state. Updating them is part of the state planning transaction,
    never a relaxation of the production drift gate.
    """
    typed_plan = deepcopy(plan.typed_plan_ir)
    for module in typed_plan.get("platform_modules", ()):
        if module.get("kind") != "state_store":
            continue
        normalize_typed_state_store_config(
            module["config"], structured_sections=plan.structured_sections,
            covers=module.get("covers", ()),
        )
        validate_platform_modules([module])
        derived = canonical_typed_state_store_config(structured_sections=sections)
        # Only authority-dependent fields change. Carry other settings verbatim
        # so deriving a new schema cannot sanitize an invalid saved contract.
        module["config"].update(schema_version=derived["schema_version"],
                                migrations=derived["migrations"])
    return typed_plan


def reconcile_behavior_state(router, plan):
    """Return a resolved plan; never mutate the saved user design in place.

    Accepted resolutions are cached under the same scope lock as behavior
    planning, with an identity containing the exact variables and behaviors.
    Every missing concept is resolved once even when many behaviors share it.
    """
    scope = deepcopy(plan.request_scope)
    behaviors = scope.get("behavior_requirements", [])
    if not behaviors:
        return plan
    sections = deepcopy(plan.structured_sections)
    section = sections.get("state_model") or {"specification": {"variables": []}}
    specification = section.get("specification")
    if not isinstance(specification, dict):
        raise TypeError("BEHAVIOR_STATE_CANONICAL_SPECIFICATION_REQUIRED")
    variables = deepcopy(specification.get("variables", []))
    original_variables = deepcopy(variables)
    validate_structured_state_section(specification)
    known = {row["name"] for row in variables}
    uses = {}
    for behavior in behaviors:
        for field in ("state_reads", "state_writes"):
            for name in behavior.get(field, ()):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("BEHAVIOR_STATE_REFERENCE_INVALID")
                if name not in known:
                    uses.setdefault(name, []).append({
                        "access": field, "behavior_id": behavior["behavior_id"],
                        "intent": behavior["intent"], "trigger": behavior.get("trigger", ""),
                        "effects": behavior["effects"], "source_quote": behavior["source_quote"],
                    })
    if not uses:
        return plan

    from .construction_behavior_checkpoint import BehaviorPlanningCheckpoint
    checkpoint = BehaviorPlanningCheckpoint(scope, router)
    identity = _digest({"schema": "mmm/behavior-state-bindings-v2",
                        "variables": variables, "behaviors": behaviors})
    path = (checkpoint.root / "state-bindings" / (identity + ".json")
            if checkpoint.enabled else None)
    with checkpoint.claim():
        stored = _read_checkpoint(path, identity) if path is not None else {}
        accepted = {name: json.loads(value) for name, value in stored.items()}
        for name in sorted(uses):
            binding = accepted.get(name)
            if binding is None:
                binding = _binding(router, name, uses[name], variables)
                accepted[name] = binding
                if path is not None:
                    _save_checkpoint(path, identity, {
                        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                        for key, value in accepted.items()})
            _validate_replayed_binding(name, binding, variables)
            if binding["binding"] == "new_scalar_state":
                variables.append(deepcopy(binding["variable"]))
                validate_structured_state_section({**specification, "variables": variables})

    for behavior in behaviors:
        for field, runtime_field in (("state_reads", "runtime_state_reads"),
                                     ("state_writes", "runtime_state_writes")):
            canonical = []
            runtime = list(behavior.get(runtime_field, ()))
            for name in behavior.get(field, ()):
                binding = accepted.get(name)
                if binding is None:
                    canonical.append(name)
                elif binding["binding"] == "world_state":
                    runtime.append(name)
                else:
                    canonical.append(binding["state_name"])
            behavior[field] = list(dict.fromkeys(canonical))
            if runtime:
                behavior[runtime_field] = list(dict.fromkeys(runtime))
    specification["variables"] = variables
    validate_structured_state_section(specification)
    sections["state_model"] = section
    scope["behavior_state_bindings"] = {**scope.get("behavior_state_bindings", {}), **accepted}
    typed_plan = (_rebind_state_stores(plan, sections)
                  if variables != original_variables else plan.typed_plan_ir)
    print("CONSTRUCTION_BEHAVIOR_STATE_BOUND: " + json.dumps({
        "references": len(accepted), "new_variables": sum(
            b["binding"] == "new_scalar_state" for b in accepted.values()),
        "runtime_observations": sum(b["binding"] == "world_state" for b in accepted.values()),
        "runtime_verified": False,
    }), flush=True)
    return replace(plan, request_scope=scope, structured_sections=sections,
                   typed_plan_ir=typed_plan)
