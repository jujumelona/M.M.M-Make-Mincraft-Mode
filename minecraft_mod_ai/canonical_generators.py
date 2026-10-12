"""Executable bounded generator for manifest-defined canonical responsibilities."""
from __future__ import annotations

import json

from .implementation_identity import compute_content_hash, compute_json_schema_hash
from .parallel_model_tasks import deterministic_model_map
from .task_template_catalog import load_template


def generate_canonical_leaf(inputs, *, leaf_id, router, authority, source_context=()):
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
        from .fixed_template_generation import _adapter_name

        slot_schema = slot["schema"]
        is_source = (
            slot_schema.get("type") == "string"
            and slot_schema.get(SCHEMA_STRING_CLASS_KEY) == STRING_CLASS_SOURCE
        )
        system_message = manifest["task"] + "\n" + "\n".join(manifest.get("rules", ()))
        if is_source:
            source_kind = "Java compilation unit" if spec.get("language") == "java" else "source artifact"
            system_message += (
                f"\nProduce a COMPLETE {source_kind}, not a placeholder, TODO, "
                "Markdown fence, or explanation. Implement the requested behavior."
            )
            if slot_schema.get("pattern"):
                system_message += (
                    "\nThe source MUST match the declared source contract: "
                    + slot_schema["pattern"]
                )
            if spec.get("language") == "java" and str(
                spec.get("bindings", {}).get("minecraft_version", "")
            ).startswith("26."):
                system_message += (
                    "\nMinecraft 26.x: use only official Mojang API names, not "
                    "obsolete Yarn net.minecraft.util.Identifier or "
                    "net.minecraft.text.Text. Bind methods to the selected target."
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
                        "materialized_prerequisite_source": source_context,
                    },
                    sort_keys=True,
                ),
            },
        ]
        # Compile full Java classes from *first-pass* bounded, complete member
        # actions. Do not repeatedly ask the model to regenerate an oversized
        # compilation unit with a larger output allowance.
        if (
            is_source
            and spec.get("language") == "java"
            and _adapter_name(router, "coder") == "llama_cpp"
        ):
            from .first_pass_java_source import (
                can_compose_first_pass_java,
                compose_first_pass_java,
            )

            if can_compose_first_pass_java(spec):
                value = compose_first_pass_java(
                    router, spec, messages, source_context=source_context,
                )
            else:
                value = generate_fixed_template_value(
                    router,
                    "coder",
                    messages,
                    response_schema=model_schemas[slot["name"]],
                    enable_tools=False,
                )
        else:
            value = generate_fixed_template_value(
                router,
                "coder",
                messages,
                response_schema=model_schemas[slot["name"]],
                enable_tools=False,
            )

        # One authored candidate only. Production checks the assembled source;
        # a failed check does not launch another full-file inference.
        validate_json_schema(value, slot_schema)
        if is_source and spec.get("language") == "java":
            from .canonical_java_target_contract import assert_canonical_java_target

            assert_canonical_java_target(value, spec)
        return value

    from .canonical_java_target_contract import assert_canonical_java_target

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
    if spec.get("language") == "java":
        assert_canonical_java_target(source, spec)

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
