from __future__ import annotations

"""Atomic design slot pipeline for small models.

Compiles high-level user prompts (e.g. '우주모드 만들어') into discrete, bounded design
slots (core loop, first goal, progression condition, reward) strictly complying with
physical atomicity bounds (MAX_MODEL_FIELDS <= 3, MAX_MODEL_STRING_CHARS <= 256).
"""

from collections.abc import Mapping
from typing import Any

from .atomic_slot_executor import SlotDefinition, fill_one_slot
from .implementation_fact import FactProvenance, FactType, ImplementationFact
from .task_template_catalog import load_template


DESIGN_SLOTS: tuple[str, ...] = (
    "design/core_loop",
    "design/first_goal",
    "design/progression_condition",
    "design/reward",
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _sanitize_title(prompt: str) -> str:
    cleaned = _text(prompt)
    if not cleaned:
        return "Custom Mod"
    tokens = [w.capitalize() for w in cleaned.split() if len(w) < 20]
    title = " ".join(tokens[:5])
    return title or "Custom Mod"


def _deterministic_slot_value(slot_id: str, prompt: str) -> str:
    p = _text(prompt).lower()
    if slot_id == "core_loop":
        if "우주" in p or "space" in p:
            return "Mine cosmic ore, refine rocket fuel, and launch into orbit."
        if "보석" in p or "gem" in p or "ruby" in p:
            return "Mine deep gems, cut gemstones on the workbench, and socket equipment."
        return f"Explore the world, gather resources for {p}, and craft advanced gear."

    if slot_id == "first_goal":
        if "우주" in p or "space" in p:
            return "Mine stardust ore from underground caverns."
        if "보석" in p or "gem" in p or "ruby" in p:
            return "Find raw gemstone ore in deep underground caves."
        return f"Collect basic materials to craft initial {p} components."

    if slot_id == "progression_condition":
        if "우주" in p or "space" in p:
            return "Collect 10 stardust ingots to construct the launchpad."
        if "보석" in p or "gem" in p or "ruby" in p:
            return "Refine 8 cut gems to unlock the infusion altar."
        return "Assemble the primary assembly station with required components."

    if slot_id == "reward":
        if "우주" in p or "space" in p:
            return "Unlock the space suit granting zero-gravity movement."
        if "보석" in p or "gem" in p or "ruby" in p:
            return "Socketed armor with enhanced durability and speed."
        return "Unlock tier-2 equipment and specialized abilities."

    return f"Completed {slot_id} objective."


def resolve_design_slot(
    router: Any,
    slot_template_id: str,
    *,
    prompt: str,
    context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Resolve a single atomic design slot, returning (slot_id, resolved_string)."""
    template = load_template(slot_template_id)
    slot_id = str(template.get("slot_id") or slot_template_id.rsplit("/", 1)[-1])
    schema = template.get("record_schema") or template.get("output_schema") or {
        "type": "object",
        "properties": {slot_id: {"type": "string", "maxLength": 256}},
        "required": [slot_id],
        "additionalProperties": False,
    }

    slot_def = SlotDefinition(
        slot_id=slot_id,
        schema=dict(schema),
        description=str(template.get("task", f"Resolve {slot_id}")),
    )
    slot_def.validate_schema()

    slot_context = {
        "prompt": prompt,
        "task": slot_def.description,
        **(dict(context) if context else {}),
    }

    if router is None:
        return slot_id, _deterministic_slot_value(slot_id, prompt)

    try:
        result = fill_one_slot(router, slot_def, slot_context, role="planner")
        if isinstance(result, Mapping) and slot_id in result:
            val = _text(result[slot_id])
            if val:
                return slot_id, val[:256]
        elif isinstance(result, str):
            val = _text(result)
            if val:
                return slot_id, val[:256]
    except Exception:
        # Fall back to deterministic grounded value if router does not support atomic decision
        pass

    return slot_id, _deterministic_slot_value(slot_id, prompt)


def compile_atomic_design(
    prompt: str,
    router: Any = None,
    *,
    research: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile discrete atomic design slots into a canonical game design specification."""
    prompt_text = _text(prompt)
    if not prompt_text:
        raise ValueError("ATOMIC_DESIGN: prompt must not be empty")

    resolved_slots: dict[str, str] = {}
    context = {"research_summary": research.get("summary", "") if research else ""}

    for slot_tmpl in DESIGN_SLOTS:
        slot_id, value = resolve_design_slot(
            router, slot_tmpl, prompt=prompt_text, context=context
        )
        resolved_slots[slot_id] = value

    core_loop_val = resolved_slots.get("core_loop") or _deterministic_slot_value("core_loop", prompt_text)
    first_goal_val = resolved_slots.get("first_goal") or _deterministic_slot_value("first_goal", prompt_text)
    prog_cond_val = (
        resolved_slots.get("progression_condition")
        or _deterministic_slot_value("progression_condition", prompt_text)
    )
    reward_val = resolved_slots.get("reward") or _deterministic_slot_value("reward", prompt_text)

    title = _sanitize_title(prompt_text)
    stem = "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")
    mod_id = f"{stem or 'mod'}_mod"

    # Derive atomic implementation facts
    facts: list[ImplementationFact] = [
        ImplementationFact(
            fact_id=f"fact_{stem}_core_item",
            fact_type=FactType.ITEM_EXISTS,
            subject=f"{stem}_core",
            value={"display_name_en": f"{title} Core"},
            provenance=FactProvenance.DESIGN,
            display_name=f"{title} Core Item",
        ),
        ImplementationFact(
            fact_id=f"fact_{stem}_core_stack",
            fact_type=FactType.ITEM_STACK_LIMIT,
            subject=f"{stem}_core",
            value=64,
            provenance=FactProvenance.DESIGN,
            display_name=f"{title} Core Stack Limit",
        ),
        ImplementationFact(
            fact_id=f"fact_{stem}_block",
            fact_type=FactType.BLOCK_EXISTS,
            subject=f"{stem}_block",
            value={"display_name_en": f"{title} Block"},
            provenance=FactProvenance.DESIGN,
            display_name=f"{title} Block",
        ),
        ImplementationFact(
            fact_id=f"fact_{stem}_block_drop",
            fact_type=FactType.BLOCK_DROP,
            subject=f"{stem}_block",
            object=f"{stem}_core",
            provenance=FactProvenance.DESIGN,
            display_name=f"{title} Block Drop",
        ),
    ]

    canonical_design: dict[str, Any] = {
        "title": title,
        "pitch": f"Minecraft mod implementing {prompt_text}.",
        "core_loop": [core_loop_val],
        "progression": [
            f"First Goal: {first_goal_val}",
            f"Condition: {prog_cond_val}",
            f"Reward: {reward_val}",
        ],
        "combat": {},
        "mod_context": {"prompt": prompt_text},
        "modules": [
            {
                "plugin_id": f"design_{mod_id}_item",
                "kind": "item",
                "status": "custom_required",
                "capability": f"{stem}_core",
                "reason": core_loop_val,
                "requirement_refs": ["req_001"],
                "implementation_obligations": [f"Register item {stem}_core", core_loop_val],
            },
            {
                "plugin_id": f"design_{mod_id}_block",
                "kind": "block",
                "status": "custom_required",
                "capability": f"{stem}_block",
                "reason": first_goal_val,
                "requirement_refs": ["req_002"],
                "implementation_obligations": [f"Register block {stem}_block", first_goal_val],
            },
        ],
        "assets": [],
        "acceptance_tests": [
            f"Core loop verified: {core_loop_val}",
            f"First goal reachable: {first_goal_val}",
            f"Progression unlocked by: {prog_cond_val}",
            f"Reward granted: {reward_val}",
        ],
        "_design_slots": resolved_slots,
        "_implementation_facts": facts,
    }
    return canonical_design
