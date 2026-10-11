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

    def generate_slot(slot, *, repair_note: str = ""):
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
            if spec.get("language") == "java" and str(
                spec.get("bindings", {}).get("minecraft_version", "")
            ).startswith("26."):
                system_message += (
                    "\nCRITICAL MINECRAFT 26.x API CONTRACT: use ONLY official "
                    "Mojang names for this bound target. NEVER import or reference "
                    "Yarn-era net.minecraft.util.Identifier, net.minecraft.text.Text, "
                    "net.minecraft.client.gui.screen.Screen, or other pre-26 "
                    "mapping namespaces. The resource identifier type is "
                    "net.minecraft.resources.ResourceLocation; text components "
                    "use net.minecraft.network.chat.Component. Verify method "
                    "signatures against the bound target, not older tutorials. "
                    "A syntactically valid obsolete Yarn import will be rejected."
                )
            if repair_note:
                system_message += (
                    "\nThe previous complete rendered Java artifact failed this "
                    "host-owned target check: " + repair_note
                    + "\nRegenerate this slot using only the bound target's "
                    "official Mojang API; preserve the stated behavior."
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
                if is_source and spec.get("language") == "java":
                    # A valid package/class declaration is not sufficient:
                    # the Minecraft 26.x target must reject obsolete Yarn API
                    # names *inside* the bounded source-repair loop. The
                    # dispatcher repeats this check before materialization.
                    from .canonical_java_target_contract import assert_canonical_java_target

                    assert_canonical_java_target(value, spec)
            except (ValidationError, ValueError) as exc:
                # Retry only candidate-level source violations. Misconfigured
                # target paths/bindings and unrelated ValueErrors are not
                # fixable by asking the model to regenerate source.
                if isinstance(exc, ValueError) and not str(exc).startswith((
                    "CANONICAL_MOJANG_API_REQUIRED:",
                    "CANONICAL_SCREEN_REGISTRATION_CLASS_INVALID:",
                )):
                    raise
                if attempt + 1 >= attempts:
                    raise
                diagnostic = exc.message if isinstance(exc, ValidationError) else str(exc)
                messages.append({
                    "role": "system",
                    "content": (
                        "The host REJECTED the previous source candidate. "
                        f"Validation error: {diagnostic}. "
                        "Regenerate the entire source string for the supplied Minecraft "
                        "target and canonical API namespace; do not reuse obsolete "
                        "mappings or guess signatures. Satisfy the exact package/type "
                        "declaration and original implementation requirement. "
                        "Do not repeat a placeholder; the host schema is unchanged."
                    ),
                })
                continue
            return value
        raise AssertionError("unreachable: canonical slot repair has no validated result")

    # Validate the COMPLETE rendered compilation unit as well as individual
    # model slots. A mold can introduce imports absent from an isolated slot.
    from .execution_contract_policy import SOURCE_REPAIR_HARD_ATTEMPTS
    from .canonical_java_target_contract import assert_canonical_java_target

    java_target = spec.get("language") == "java"
    attempts = max(1, SOURCE_REPAIR_HARD_ATTEMPTS) if java_target else 1
    repair_note = ""
    for render_attempt in range(attempts):
        generated = deterministic_model_map(
            router,
            slots,
            lambda slot: generate_slot(slot, repair_note=repair_note),
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
        if java_target:
            try:
                assert_canonical_java_target(source, spec)
            except ValueError as exc:
                if not str(exc).startswith((
                    "CANONICAL_MOJANG_API_REQUIRED:",
                    "CANONICAL_SCREEN_REGISTRATION_CLASS_INVALID:",
                )) or render_attempt + 1 >= attempts or not slots:
                    raise
                repair_note = str(exc)
                continue
        break
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
