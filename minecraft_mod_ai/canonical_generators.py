"""Executable bounded generator for manifest-defined canonical responsibilities."""
from __future__ import annotations

import json

from .implementation_identity import compute_content_hash, compute_json_schema_hash
from .parallel_model_tasks import deterministic_model_map
from .task_template_catalog import load_template


def generate_canonical_leaf(inputs, *, leaf_id, router, authority):
    """Generate one declared artifact, keeping identity/receipt fields host-owned."""
    from .fixed_template_generation import generate_fixed_template_value
    from .integrity_validators import validate_semantic_contract

    manifest = load_template(leaf_id)
    authority.types.validate_input(f"{leaf_id}:input", inputs)
    spec_port = next(p["name"] for p in manifest["inputs"] if p["type"] == "specification")
    spec = inputs[spec_port]
    if router is None and spec["slots"]:
        raise ValueError("CANONICAL_GENERATOR_ROUTER_REQUIRED")
    from .implementation_template_renderer import render_template
    from .model_output_atomicity_contract import assert_strict_atomicity_bounds

    values = dict(spec["bindings"])
    slots = tuple(spec["slots"])
    for slot in slots:
        name = slot["name"]
        if name in values:
            raise ValueError("GENERATOR_SLOT_BINDING_CONFLICT")
        assert_strict_atomicity_bounds(
            slot["schema"], surface=f"canonical slot {leaf_id}:{name}"
        )

    def generate_slot(slot):
        return generate_fixed_template_value(
            router,
            "coder",
            [
                {
                    "role": "system",
                    "content": manifest["task"] + "\n" + "\n".join(manifest.get("rules", ())),
                },
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
            ],
            response_schema=slot["schema"],
            enable_tools=False,
        )

    generated = deterministic_model_map(
        router,
        slots,
        generate_slot,
        role="coder",
        thread_name_prefix="canonical-slot",
    )
    for slot, value in zip(slots, generated):
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
