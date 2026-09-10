from __future__ import annotations

"""Atomic design slot pipeline for small models.

Compiles user requests and grounded research into discrete, bounded design
slots strictly complying with physical atomicity bounds (MAX_MODEL_FIELDS <= 3,
MAX_MODEL_STRING_CHARS <= 256, MAX_SCHEMA_DEPTH <= 2).
"""

from collections.abc import Mapping, Sequence
import hashlib
import re
from typing import Any

from .atomic_slot_executor import SlotDefinition, SlotFillError, fill_one_slot
from .complete_spec import AssetRequest, ProductionModule
from .implementation_fact import FactProvenance, FactType, ImplementationFact
from .task_template_catalog import load_template

# Full ontology of 32 atomic design slots from high-level fantasy down to mechanics, systems, progression, and assets
ALL_DESIGN_SLOTS: tuple[str, ...] = (
    "design/audio_identity",
    "design/combat_role",
    "design/content_scale",
    "design/core_action",
    "design/core_loop",
    "design/core_loop_step",
    "design/crafting_role",
    "design/economy_sink",
    "design/economy_source",
    "design/enemy_role",
    "design/exploration_target",
    "design/failure_condition",
    "design/first_goal",
    "design/goal_prerequisite",
    "design/machine_role",
    "design/network_requirement",
    "design/npc_role",
    "design/persistence_requirement",
    "design/player_fantasy",
    "design/progression_condition",
    "design/progression_edge",
    "design/progression_node",
    "design/resource_sink",
    "design/resource_source",
    "design/reward",
    "design/risk",
    "design/texture_requirement",
    "design/theme",
    "design/ui_requirement",
    "design/unlock",
    "design/visual_identity",
    "design/world_interaction",
)
# Foundation slots always resolved for every mod
FOUNDATION_SLOTS: tuple[str, ...] = (
    "design/theme",
    "design/player_fantasy",
    "design/visual_identity",
    "design/core_action",
    "design/core_loop",
    "design/first_goal",
    "design/texture_requirement",
)

# Domain-specific child slots triggered dynamically by content domains
DOMAIN_SLOTS: dict[str, tuple[str, ...]] = {
    "item": (
        "design/resource_source",
        "design/resource_sink",
        "design/crafting_role",
        "design/reward",
    ),
    "block": (
        "design/world_interaction",
    ),
    "machine": (
        "design/machine_role",
    ),
    "combat": (
        "design/combat_role",
        "design/risk",
    ),
    "entity": (
        "design/enemy_role",
        "design/npc_role",
    ),
    "worldgen": (
        "design/exploration_target",
    ),
    "gui": (
        "design/ui_requirement",
    ),
}

PROGRESSION_SLOTS: tuple[str, ...] = (
    "design/progression_condition",
    "design/reward",
)

CORE_DESIGN_SLOTS = FOUNDATION_SLOTS
DESIGN_SLOTS = ALL_DESIGN_SLOTS


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _sanitize_stem(name: str) -> str:
    cleaned = "".join(c if (c.isascii() and c.isalnum()) else "_" for c in name.lower())
    parts = [p for p in cleaned.split("_") if p]
    stem = "_".join(parts)
    if not stem:
        stem = f"mmm_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:10]}"
    if not stem[0].isalpha() or not stem[0].isascii():
        stem = f"mod_{stem}"
    return stem[:30]


def _format_research_context(research: Mapping[str, Any] | None) -> str:
    if not research:
        return ""
    lines: list[str] = []
    summary = _text(research.get("summary"))
    if summary:
        lines.append(f"Summary: {summary}")
    claims = research.get("claims") or ()
    if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes)):
        for claim in claims[:4]:
            t = _text(claim)
            if t:
                lines.append(f"- Claim: {t}")
    refs = research.get("references") or ()
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        for ref in refs[:3]:
            t = _text(ref)
            if t:
                lines.append(f"- Reference: {t}")
    known = research.get("known") or ()
    if isinstance(known, Sequence) and not isinstance(known, (str, bytes)):
        for k in known[:3]:
            t = _text(k)
            if t:
                lines.append(f"- Known: {t}")
    return "\n".join(lines)[:2048]


def _deterministic_prompt_fallback(slot_id: str, prompt: str, research_context: str) -> str:
    """Derive a bounded slot value directly from prompt and research text without hardcoded topic keywords."""
    clean_p = _text(prompt)
    lead = clean_p[:120] if clean_p else "Custom mod functionality"

    if slot_id == "theme":
        return f"Theme focused on {lead}."
    if slot_id == "player_fantasy":
        return f"Player engages directly with {lead} in Minecraft."
    if slot_id == "visual_identity":
        return f"Distinctive pixel styling appropriate for {lead}."
    if slot_id == "core_action":
        return f"Interact with and utilize mechanics of {lead}."
    if slot_id == "core_loop_step":
        return f"Gather supplies, apply {lead}, and progress through gameplay tiers."
    if slot_id == "core_loop":
        return f"Engage with {lead} to build and advance abilities."
    if slot_id == "resource_source":
        return f"Acquire resources required for {lead} from the game world."
    if slot_id == "resource_sink":
        return f"Consume materials to sustain and upgrade {lead}."
    if slot_id == "first_goal":
        return f"Obtain the primary starter item or block for {lead}."
    if slot_id == "goal_prerequisite":
        return f"Collect basic Minecraft resources to initiate {lead}."
    if slot_id == "progression_condition":
        return f"Assemble or reach key milestones for {lead}."
    if slot_id == "reward":
        return f"Earn capabilities and equipment granted by {lead}."
    if slot_id == "unlock":
        return f"Unlock subsequent tiers of {lead}."
    if slot_id == "failure_condition":
        return f"Depleting resources or failing challenges associated with {lead}."
    if slot_id == "texture_requirement":
        stem = _sanitize_stem(clean_p)
        return f"{stem}_primary icon and {stem}_base texture."
    if slot_id == "content_scale":
        return f"Standard mod scope implementing {lead}."

    return f"Specific {slot_id} implementation for {lead}."


def resolve_design_slot(
    router: Any,
    slot_template_id: str,
    *,
    prompt: str,
    research_context: str = "",
) -> tuple[str, str]:
    """Resolve a single atomic design slot with fail-closed schema validation."""
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
        "research": research_context,
    }

    if router is None:
        return slot_id, _deterministic_prompt_fallback(slot_id, prompt, research_context)

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
    except SlotFillError:
        raise
    except Exception as exc:
        raise SlotFillError(
            f"SLOT_RESOLUTION_FAILED: Failed to resolve {slot_id} for prompt {prompt!r}: {exc}"
        ) from exc

    return slot_id, _deterministic_prompt_fallback(slot_id, prompt, research_context)


def _fallback_content_domains(prompt: str) -> list[str]:
    """Fallback classification when router is unavailable or fails."""
    p = prompt.lower()
    domains: list[str] = []
    if any(w in p for w in ("block", "블록", "광석", "ore", "tile")):
        domains.append("block")
    if any(w in p for w in ("entity", "mob", "몬스터", "생물", "creature", "boss", "보스")):
        domains.append("entity")
    if any(w in p for w in ("machine", "기계", "장치")):
        domains.append("machine")
    if any(w in p for w in ("combat", "전투", "무기", "weapon", "sword", "검")):
        domains.append("combat")
    if any(w in p for w in ("worldgen", "dimension", "차원", "biome", "바이옴", "우주", "space")):
        domains.append("worldgen")
    if any(w in p for w in ("gui", "ui", "hud", "화면", "메뉴")):
        domains.append("gui")
    if any(w in p for w in ("item", "아이템", "도구", "tool", "material", "재료")) or not domains:
        domains.insert(0, "item")
    return list(dict.fromkeys(domains))[:3]


def resolve_content_domains(
    router: Any,
    *,
    prompt: str,
    research_context: str = "",
) -> list[str]:
    """Resolve which Minecraft content domains are required using the model or prompt fallback."""
    if router is not None:
        try:
            template = load_template("design/content_domains")
            schema = template.get("record_schema") or template.get("output_schema")
            slot_def = SlotDefinition(
                slot_id="domains",
                schema=dict(schema),
                description=str(template.get("task", "Select content domains")),
            )
            slot_def.validate_schema()
            slot_context = {
                "prompt": prompt,
                "task": slot_def.description,
                "research": research_context,
            }
            result = fill_one_slot(router, slot_def, slot_context, role="planner")
            if isinstance(result, Mapping) and "domains" in result:
                domains = result["domains"]
                if isinstance(domains, list) and domains:
                    valid = [str(d).lower() for d in domains if str(d).lower() in DOMAIN_SLOTS]
                    if valid:
                        return valid[:3]
            elif isinstance(result, list):
                valid = [str(d).lower() for d in result if str(d).lower() in DOMAIN_SLOTS]
                if valid:
                    return valid[:3]
        except Exception:
            pass
    return _fallback_content_domains(prompt)


def _extract_identifiers(text: str, default_stem: str) -> list[str]:
    """Extract clean identifier tokens from a design slot text."""
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,20}\b", text.lower())
    stopwords = {
        "define", "primary", "minecraft", "custom", "player", "mechanic", "gameplay",
        "action", "gather", "craft", "obtain", "reward", "unlock", "system", "icon",
        "texture", "from", "with", "into", "this", "that", "world", "items", "blocks",
        "needed", "appropriate", "distinctive", "styling"
    }
    filtered = [t for t in tokens if t not in stopwords]
    return filtered or [default_stem]


def compile_atomic_design(
    prompt: str,
    router: Any = None,
    *,
    research: Mapping[str, Any] | None = None,
    request_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile discrete atomic design slots into an authoritative canonical game design specification."""
    prompt_text = _text(prompt)
    if not prompt_text:
        raise ValueError("ATOMIC_DESIGN: prompt must not be empty")

    research_context = _format_research_context(research)
    resolved_slots: dict[str, str] = {}

    # 1. Foundation slots
    active_slots: list[str] = list(FOUNDATION_SLOTS)

    # 2. Dynamic content domains (model-classified or fallback)
    domains = resolve_content_domains(
        router,
        prompt=prompt_text,
        research_context=research_context,
    )
    for domain in domains:
        for s in DOMAIN_SLOTS.get(domain, ()):
            if s not in active_slots:
                active_slots.append(s)

    # 3. Progression condition
    for s in PROGRESSION_SLOTS:
        if s not in active_slots:
            active_slots.append(s)

    # Resolve active slots dynamically
    for slot_tmpl in active_slots:
        slot_id, value = resolve_design_slot(
            router,
            slot_tmpl,
            prompt=prompt_text,
            research_context=research_context,
        )
        resolved_slots[slot_id] = value

    # Populate all remaining slots with deterministic bounded fallbacks so downstream never sees missing keys
    for slot_tmpl in ALL_DESIGN_SLOTS:
        s_id = slot_tmpl.rsplit("/", 1)[-1]
        if s_id not in resolved_slots:
            resolved_slots[s_id] = _deterministic_prompt_fallback(s_id, prompt_text, research_context)

    stem = _sanitize_stem(prompt_text)
    mod_id = f"{stem}_mod"
    title = prompt_text.capitalize() if len(prompt_text) <= 40 else f"{stem.replace('_', ' ').title()} Mod"

    core_loop_val = resolved_slots.get("core_loop") or _deterministic_prompt_fallback("core_loop", prompt_text, research_context)
    first_goal_val = resolved_slots.get("first_goal") or _deterministic_prompt_fallback("first_goal", prompt_text, research_context)
    prog_cond_val = resolved_slots.get("progression_condition") or _deterministic_prompt_fallback("progression_condition", prompt_text, research_context)
    reward_val = resolved_slots.get("reward") or _deterministic_prompt_fallback("reward", prompt_text, research_context)
    visual_identity_val = resolved_slots.get("visual_identity") or f"Pixel art styling for {stem}"
    texture_req_val = resolved_slots.get("texture_requirement") or f"{stem}_item and {stem}_block textures"

    # Derive requirements from request_catalog or active requirement ledger
    from .design_requirement_contract import _active_requirement_ledger
    ledger = ()
    if request_catalog and isinstance(request_catalog.get("requirements"), list):
        ledger = tuple(request_catalog["requirements"])
    else:
        ledger = _active_requirement_ledger(prompt_text)

    all_req_ids = [
        str(r.get("requirement_id")).strip()
        for r in ledger
        if str(r.get("requirement_id")).strip()
    ]

    facts: list[ImplementationFact] = []
    modules: list[ProductionModule] = []
    assets: list[AssetRequest] = []

    has_item = "item" in domains
    has_block = "block" in domains
    if not (has_item or has_block):
        has_item = True

    # Dynamic Items
    if has_item:
        item_id = f"{stem}_item"
        req_refs = list(all_req_ids) if all_req_ids else [f"req_{item_id}"]
        item_obligations = [f"Register item {item_id}", first_goal_val]
        if ledger:
            for r in ledger:
                st = _text(r.get("statement") or r.get("semantic_statement"))
                if st and st not in item_obligations:
                    item_obligations.append(st)
        facts.append(
            ImplementationFact(
                fact_id=f"fact_{item_id}_exists",
                fact_type=FactType.ITEM_EXISTS,
                subject=item_id,
                value={"display_name_en": f"{item_id.replace('_', ' ').title()}"},
                provenance=FactProvenance.DESIGN,
                display_name=f"{item_id.replace('_', ' ').title()}",
            )
        )
        facts.append(
            ImplementationFact(
                fact_id=f"fact_{item_id}_stack",
                fact_type=FactType.ITEM_STACK_LIMIT,
                subject=item_id,
                value=64,
                provenance=FactProvenance.DESIGN,
                display_name=f"{item_id} Stack Limit",
            )
        )
        modules.append(
            ProductionModule(
                module_id=item_id,
                kind="item",
                config={
                    "item_id": item_id,
                    "name": f"{item_id.replace('_', ' ').title()}",
                    "plugin_id": item_id,
                    "requirement_refs": req_refs,
                    "implementation_obligations": item_obligations,
                    "reason": first_goal_val,
                },
                depends_on=(),
                required_gates=(),
            )
        )
        assets.append(
            AssetRequest(
                asset_id=f"texture_item_{item_id}",
                kind="item",
                target_path=f"assets/{mod_id}/textures/item/{item_id}.png",
                width=16,
                height=16,
                prompt=f"Pixel Art, PixArFK, {visual_identity_val}, {texture_req_val}, {item_id.replace('_', ' ')} item icon, isolated transparent background",
            )
        )

    # Dynamic Blocks
    if has_block:
        block_id = f"{stem}_block"
        drop_target = f"{stem}_item" if has_item else block_id
        req_refs = list(all_req_ids) if all_req_ids else [f"req_{block_id}"]
        block_obligations = [f"Register block {block_id}", core_loop_val]
        if ledger:
            for r in ledger:
                st = _text(r.get("statement") or r.get("semantic_statement"))
                if st and st not in block_obligations:
                    block_obligations.append(st)
        facts.append(
            ImplementationFact(
                fact_id=f"fact_{block_id}_exists",
                fact_type=FactType.BLOCK_EXISTS,
                subject=block_id,
                value={"display_name_en": f"{block_id.replace('_', ' ').title()}"},
                provenance=FactProvenance.DESIGN,
                display_name=f"{block_id.replace('_', ' ').title()}",
            )
        )
        facts.append(
            ImplementationFact(
                fact_id=f"fact_{block_id}_drop",
                fact_type=FactType.BLOCK_DROP,
                subject=block_id,
                object=drop_target,
                provenance=FactProvenance.DESIGN,
                display_name=f"{block_id} Drop",
            )
        )
        modules.append(
            ProductionModule(
                module_id=block_id,
                kind="block",
                config={
                    "block_id": block_id,
                    "name": f"{block_id.replace('_', ' ').title()}",
                    "plugin_id": block_id,
                    "requirement_refs": req_refs,
                    "implementation_obligations": block_obligations,
                    "reason": core_loop_val,
                    "drop": drop_target,
                },
                depends_on=(),
                required_gates=(),
            )
        )
        assets.append(
            AssetRequest(
                asset_id=f"texture_block_{block_id}",
                kind="block",
                target_path=f"assets/{mod_id}/textures/block/{block_id}.png",
                width=16,
                height=16,
                prompt=f"Pixel Art, PixArFK, {visual_identity_val}, {texture_req_val}, {block_id.replace('_', ' ')} block texture, seamless square tile",
            )
        )

    # Acceptance tests derived strictly from design slots
    acceptance_tests = [
        f"Core loop verified: {core_loop_val}",
        f"First goal reachable: {first_goal_val}",
        f"Progression unlocked by: {prog_cond_val}",
        f"Reward granted: {reward_val}",
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
        "combat": {"role": resolved_slots.get("combat_role", "none")},
        "mod_context": {
            "prompt": prompt_text,
            "theme": resolved_slots.get("theme", ""),
            "player_fantasy": resolved_slots.get("player_fantasy", ""),
        },
        "modules": modules,
        "assets": assets,
        "acceptance_tests": acceptance_tests,
        "_design_slots": resolved_slots,
        "_implementation_facts": facts,
        "_content_domains": domains,
    }
    return canonical_design


__all__ = [
    "ALL_DESIGN_SLOTS",
    "CORE_DESIGN_SLOTS",
    "DESIGN_SLOTS",
    "compile_atomic_design",
    "resolve_design_slot",
]
