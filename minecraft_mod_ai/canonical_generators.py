"""Executable bounded generator for manifest-defined canonical responsibilities."""
from __future__ import annotations

import json

from .implementation_identity import compute_content_hash, compute_json_schema_hash
from .parallel_model_tasks import deterministic_model_map
from .task_template_catalog import load_template


def generate_canonical_leaf(inputs, *, leaf_id, router, authority):
    """Generate one declared artifact, keeping identity/receipt fields host-owned."""
    from .fixed_template_generation import generate_fixed_template_value
    from .integrity_validators import validate_semantic_contract, validate_json_schema

    manifest = load_template(leaf_id)
    authority.types.validate_input(f"{leaf_id}:input", inputs)
    spec_port = next(p["name"] for p in manifest["inputs"] if p["type"] == "specification")
    spec = inputs[spec_port]
    if router is None and spec["slots"]:
        raise ValueError("CANONICAL_GENERATOR_ROUTER_REQUIRED")
    from .implementation_template_renderer import render_template
    from .model_output_atomicity_contract import assert_atomic_model_schema

    from .execution_contract_policy import SCHEMA_STRING_CLASS_KEY, STRING_CLASS_SOURCE

    values = dict(spec["bindings"])
    slots = tuple(spec["slots"])
    model_schemas = {}
    for slot in slots:
        name = slot["name"]
        if name in values:
            raise ValueError("GENERATOR_SLOT_BINDING_CONFLICT")
        schema = dict(slot["schema"])
        if schema.get("type") == "string" and schema.get(SCHEMA_STRING_CLASS_KEY) == STRING_CLASS_SOURCE:
            # Package/class/lifecycle regexes are host proof obligations, not
            # llama.cpp grammar. Keep the persisted contract intact and check
            # the returned slot against it before rendering or publication.
            schema.pop("pattern", None)
        assert_atomic_model_schema(schema, surface=f"canonical slot {leaf_id}:{name}")
        model_schemas[name] = schema

    def generate_slot(slot):
        from jsonschema.exceptions import ValidationError

        from .execution_contract_policy import SOURCE_REPAIR_HARD_ATTEMPTS

        slot_schema = slot["schema"]
        is_source = (
            slot_schema.get("type") == "string"
            and slot_schema.get(SCHEMA_STRING_CLASS_KEY) == STRING_CLASS_SOURCE
        )
        system_message = manifest["task"] + "\n" + "\n".join(manifest.get("rules", ()))
        if is_source:
            # The llama.cpp transport cannot enforce the host's package/class regex.
            # Give the coder the exact obligation rather than allowing arbitrary
            # scalar text to pass until a later artifact-graph boundary.
            source_kind = "Java compilation unit" if spec.get("language") == "java" else "source artifact"
            system_message += (
                f"\nProduce a COMPLETE {source_kind} as the string value. "
                "Implement the stated requirement. Never return a placeholder, TODO, "
                "example label, Markdown fence, or explanation instead of source."
            )
            if slot_schema.get("pattern"):
                system_message += (
                    "\nThe complete source MUST match this host-owned source "
                    f"declaration contract: {slot_schema['pattern']}"
                )

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "identity": inputs[manifest["inputs"][0]["name"]],
                        "requirement": spec["requirement"],
                        "slot": slot["description"],
                        "bindings": spec["bindings"],
                    },
                    sort_keys=True,
                ),
            },
        ]
        # Source-only repair is bounded and never changes the host schema or
        # invents code. Every candidate must validate before the graph can
        # publish a receipt, write a file, or cache the result.
        attempts = max(1, SOURCE_REPAIR_HARD_ATTEMPTS) if is_source else 1
        for attempt in range(attempts):
            value = generate_fixed_template_value(
                router,
                "coder",
                messages,
                response_schema=model_schemas[slot["name"]],
                enable_tools=False,
            )
            try:
                validate_json_schema(value, slot_schema)
            except ValidationError as exc:
                if attempt + 1 >= attempts:
                    raise
                messages.append({
                    "role": "system",
                    "content": (
                        "The host REJECTED the previous source candidate. "
                        f"Validation error: {exc.message}. "
                        "Regenerate the entire source string, satisfying the exact "
                        "package/type declaration and original implementation requirement. "
                        "Do not repeat a placeholder; the host schema is unchanged."
                    ),
                })
                continue
            return value
        raise AssertionError("unreachable: canonical slot repair has no validated result")

    generated = deterministic_model_map(
        router,
        slots,
        generate_slot,
        role="coder",
        thread_name_prefix="canonical-slot",
    )
    for slot, value in zip(slots, generated):
        validate_json_schema(value, slot["schema"])
        values[slot["name"]] = value

    source = render_template({"render": spec["render_mold"]}, values)
    validate_semantic_contract(
        source, contract=spec, context_id=spec["context_id"], leaf_id=leaf_id
    )
    result = {}
    for port in manifest["outputs"]:
        if port["type"] == "code_fragment":
            result[port["name"]] = source
        elif port["type"] == "receipt":
            result[port["name"]] = {
                "leaf_id": leaf_id,
                "context_id": spec["context_id"],
                "content_sha256": compute_content_hash(source.encode()),
                "contract_sha256": compute_json_schema_hash(spec),
            }
        else:
            raise ValueError("UNSUPPORTED_CANONICAL_OUTPUT")
    authority.types.validate_output(f"{leaf_id}:output", result)
    return result
