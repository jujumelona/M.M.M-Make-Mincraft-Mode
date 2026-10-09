"""Pre-authored Debug input. No planner/model call is made to construct it.

This is a realistic *simulated* authored plan, not a claim that a local model
or a donor repository has produced or verified these choices. Production still
owns platform binding, source-reuse inspection/proofs and every build gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .authored_plan import AuthoredPlan
from .authored_reuse_bridge import authored_capability_graph
from .authored_structured_design import normalize_structured_sections
from .implementation_fact import ImplementationFact
from .planning_detail_slots import DETAIL_RECORDS
from .typed_host_capabilities import typed_host_capability_contracts
from .typed_plan_ir import validate_typed_plan_ir
from .typed_plan_support import assert_typed_plan_host_support
from .authored_content_contract import content_owned_refs


DEBUG_PREBUILT_PROMPT = (
    "Fabric Minecraft mod: register a crystal_fragment item and crystal_block, "
    "add a shaped 2x2 crystal_fragment crafting recipe producing crystal_block, "
    "provide item/block resources and verify gameplay and packaging."
)

DEBUG_PREBUILT_TEXT = """# Crystal crafting – pre-authored production-debug scenario
## Minecraft gameplay
1. Register crystal_fragment as an inventory item and crystal_block as a placeable block.
2. Craft one crystal_block from four crystal_fragments in a 2×2 shaped recipe.
3. Generate the item/block registry, models, localization and recipe through the
   canonical content artifact graph, not through a bypassed fixture source writer.
## Source-reuse selection
Search for Fabric item, block and crafting-recipe examples. Candidates are
reference-only until host inspection, license checks and target-specific proof.
An unverified repository is NEVER treated as a compilable donor.
## Acceptance
Confirm item and block registry identities; confirm the crafting data and its
dependencies; run the normal source, Java, Gradle, packaging and runtime gates.
Failures must remain visible as failures.
"""


def _records(section: str, active: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    """Build only fields from the canonical design schema, not ad-hoc keys."""
    active = active or {}
    spec: dict[str, Any] = {
        concern: [] for concern in DETAIL_RECORDS[section]
    }
    for concern, values in active.items():
        if concern not in spec:
            raise ValueError(f"DEBUG_PREBUILT_UNKNOWN_CONCERN: {section}.{concern}")
        spec[concern] = [{
            field: values.get(
                field,
                f"{section}.{concern}: crystal crafting fixture; verify the implementation.",
            )[:512]
            for field in DETAIL_RECORDS[section][concern].split()
        }]
    spec["inapplicable_concerns"] = []
    return {"specification": spec, "constraint_evidence_refs": []}


def build_prebuilt_debug_plan() -> AuthoredPlan:
    """Return a complete saved-plan ABI with concrete multi-kind artifact facts."""
    structured = normalize_structured_sections({
        "resources_and_ui": _records(
            "resources_and_ui",
            {
                "assets": {
                    "purpose": "crystal_fragment item and crystal_block visual models",
                    "production_owner": "crystal_fragment and crystal_block registries",
                },
                "registries": {
                    "name": "crystal_fragment and crystal_block registry identities",
                },
            },
        ),
        "reuse_assessment": _records(
            "reuse_assessment",
            {
                "adaptations": {
                    "part": "minecraft fabric item block crafting recipe registration",
                },
            },
        ),
        "verification": _records("verification"),
    })
    source_hash = "sha256:" + hashlib.sha256(
        DEBUG_PREBUILT_TEXT.encode("utf-8")
    ).hexdigest()
    typed_ir = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": source_hash,
        "functions": [],
        "initialize": [],
        "platform_modules": [],
        "event_bindings": [],
    }
    validate_typed_plan_ir(typed_ir, capabilities=typed_host_capability_contracts())

    module_specs = (
        ("crystal_fragment", "item", {"display_name": "Crystal Fragment", "stack_limit": 64}),
        ("crystal_block", "block", {"display_name": "Crystal Block", "hardness": 2.0, "resistance": 3.0}),
        ("crystal_block_recipe", "recipe", {"recipe_kind": "shaped", "result": "crystal_block"}),
    )
    modules = [
        {
            "module_id": module_id,
            "kind": kind,
            "config": config,
            "depends_on": (
                ["crystal_fragment", "crystal_block"]
                if kind == "recipe" else []
            ),
            "required_gates": [],
        }
        for module_id, kind, config in module_specs
    ]
    facts = [
        ImplementationFact(
            fact_id="debug.crystal_fragment.exists",
            fact_type="ITEM_EXISTS",
            subject="crystal_fragment",
            display_name="Crystal Fragment",
            source_clause="Register crystal_fragment as an inventory item.",
        ).to_dict(),
        ImplementationFact(
            fact_id="debug.crystal_block.exists",
            fact_type="BLOCK_EXISTS",
            subject="crystal_block",
            display_name="Crystal Block",
            source_clause="Register a placeable crystal_block.",
        ).to_dict(),
        ImplementationFact(
            fact_id="debug.crystal_block.recipe",
            fact_type="CRAFTING_RECIPE",
            subject="crystal_block_recipe",
            value={
                "kind": "shaped",
                "pattern": ["FF", "FF"],
                "key": {"F": "crystal_fragment"},
                "result_id": "crystal_block",
                "count": 1,
            },
            source_clause="Craft crystal_block using four crystal_fragment items.",
        ).to_dict(),
    ]
    graph = authored_capability_graph(DEBUG_PREBUILT_PROMPT, structured)
    content_design = {
        "_mod_id": "mmm_debug_crystal",
        "modules": modules,
        "assets": [],
        "_implementation_facts": facts,
        "_debug_reference_selection": {
            "policy": "reference_only_until_host_inspection_and_proof",
            "search_terms": graph["search_terms"],
            "verification_stage": "production_after_target_binding",
            "donor_claims": [],
            "reference_candidates": [
                {
                    "repository": "FabricMC/fabric-example-mod",
                    "selection_role": "reference_only",
                    "purpose": "Example Fabric item/block registration; compatibility must be proven",
                }
            ],
        },
        "acceptance_tests": [
            "Item and block registration are generated and loadable.",
            "Crafting recipe produces crystal_block from four crystal_fragments.",
            "Build, Java diagnostics, packaging and game gates report their actual outcomes.",
        ],
    }
    assert_typed_plan_host_support(
        structured, typed_ir,
        externally_covered_refs=content_owned_refs(structured, content_design),
    )
    plan = AuthoredPlan(
        requested_prompt=DEBUG_PREBUILT_PROMPT,
        text=DEBUG_PREBUILT_TEXT,
        structured_sections=structured,
        typed_plan_ir=typed_ir,
        content_design=content_design,
    )
    return plan


def write_prebuilt_debug_plan(target: str | Path) -> Path:
    """Persist before loading; never call session.plan or depend on network."""
    plan = build_prebuilt_debug_plan()
    destination = Path(target).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.prebuilt-{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        restored = AuthoredPlan.from_dict(
            json.loads(temporary.read_text(encoding="utf-8"))
        )
        if restored.calculate_hash() != plan.calculate_hash():
            raise RuntimeError("DEBUG_PREBUILT_PLAN_ROUNDTRIP_MISMATCH")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = ["DEBUG_PREBUILT_PROMPT", "build_prebuilt_debug_plan", "write_prebuilt_debug_plan"]
