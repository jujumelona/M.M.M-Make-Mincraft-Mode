from __future__ import annotations

from pathlib import Path
import re

SOURCE = Path("minecraft_mod_ai/agentic_research_game_design.py")
TEST = Path("tests/test_agentic_design_requirement_integrity.py")

text = SOURCE.read_text(encoding="utf-8")

production_pattern = re.compile(
    r"_PRODUCTION_DEPTH = \(\n.*?\n\)\n\n_MODULE_FORMAT = \(",
    re.S,
)
production_replacement = '''_PRODUCTION_DEPTH = (
    "PRODUCTION DEPTH: finish the game/mod design before implementation search. "
    "Decompose every requested mechanic into the smallest meaningful subsystems that can "
    "be independently implemented, tested, and searched for reuse. Split different player "
    "verbs, resources, state transitions, purchase/assembly steps, upgrade gates, travel "
    "phases, encounters, combat outcomes, world interactions, persistence-visible state, "
    "networking/client surfaces, and integration rules when they can fail independently. "
    "Do not collapse an epic such as planet interaction, ship construction, trading, or "
    "progression into one generic subsystem. Use as many concrete subsystems as the authored "
    "design genuinely needs; never add unrelated features. Use supplied research evidence "
    "for Minecraft/Fabric facts and unresolved assumptions, but donor/reuse selection happens "
    "only after this design is frozen. Do not invent target-specific APIs, storage locations, "
    "client/server authority, or implementation facts that are not supported by the supplied "
    "research; describe behavior and authority requirements instead."
)

_MODULE_PRODUCTION_DEPTH = (
    "MODULE LEAF INDEX: every implementation-bearing core-loop/progression/combat/mod-context "
    "behavior must have a concrete modules row. Preserve only exact host-approved requirement "
    "IDs in requirement_refs; never infer, synthesize, rename, or extend requirement IDs."
)

_MODULE_FORMAT = ('''
text, count = production_pattern.subn(production_replacement, text, count=1)
if count != 1:
    raise SystemExit(f"production-depth replacement count={count}")

none_anchor = '_NONE_VALUES = frozenset({"none", "n/a", "없음"})\n\n'
helper = '''_NONE_VALUES = frozenset({"none", "n/a", "없음"})
_REQUIREMENT_ID_RE = re.compile(r"\\breq_[A-Za-z0-9_]+\\b")


def _referenced_requirement_ids(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        refs.update(_REQUIREMENT_ID_RE.findall(value))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            refs.update(_referenced_requirement_ids(str(key)))
            refs.update(_referenced_requirement_ids(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            refs.update(_referenced_requirement_ids(child))
    return refs


def _assert_known_requirement_ids(
    field: str,
    value: Any,
    approved_requirement_ids: set[str],
) -> None:
    if not approved_requirement_ids:
        return
    unknown = sorted(_referenced_requirement_ids(value) - approved_requirement_ids)
    if unknown:
        raise SpecValidationError(
            f"{field} cites unknown requirement ids: " + ", ".join(unknown)
        )

'''
if none_anchor not in text:
    raise SystemExit("requirement helper anchor missing")
text = text.replace(none_anchor, helper, 1)

# Patch only the acceptance validator, never the Markdown parser that has a similar
# assert_design_field_clean sequence.
validate_start = text.index("def _validate_section_types(")
validate_end = text.index("\ndef _validate_requirement_coverage(", validate_start)
validate_block = text[validate_start:validate_end]
validation_anchor = '''        try:
            assert_design_field_clean(field, value)
'''
validation_replacement = '''        _assert_known_requirement_ids(field, value, required_ids)
        try:
            assert_design_field_clean(field, value)
'''
if validate_block.count(validation_anchor) != 1:
    raise SystemExit(
        f"validate-section clean-boundary count={validate_block.count(validation_anchor)}"
    )
validate_block = validate_block.replace(validation_anchor, validation_replacement, 1)
text = text[:validate_start] + validate_block + text[validate_end:]

old_section_rule = '            "Preserve exact approved requirement IDs in modules. "\n'
new_section_rule = '            "Never invent requirement IDs; when citing one, use only an exact host-approved ID. "\n'
if old_section_rule not in text:
    raise SystemExit("section requirement rule anchor missing")
text = text.replace(old_section_rule, new_section_rule, 1)

section_depth_anchor = '''            + _PRODUCTION_DEPTH
        )
        ledger = _active_requirement_ledger(prompt)
'''
section_depth_replacement = '''            + _PRODUCTION_DEPTH
            + (" " + _MODULE_PRODUCTION_DEPTH if "modules" in fields else "")
        )
        ledger = _active_requirement_ledger(prompt)
'''
if section_depth_anchor not in text:
    raise SystemExit("section depth anchor missing")
text = text.replace(section_depth_anchor, section_depth_replacement, 1)

old_field_rule = '        "code fences, <think>, analysis, or unrelated fields. Preserve exact approved requirement IDs when requested. "\n'
new_field_rule = '        "code fences, <think>, analysis, or unrelated fields. Never invent requirement IDs; cite only exact host-approved IDs. "\n'
if old_field_rule not in text:
    raise SystemExit("field requirement rule anchor missing")
text = text.replace(old_field_rule, new_field_rule, 1)

field_depth_anchor = '''        + _PRODUCTION_DEPTH
    )
    ledger = _active_requirement_ledger(prompt)
'''
field_depth_replacement = '''        + _PRODUCTION_DEPTH
        + (" " + _MODULE_PRODUCTION_DEPTH if field == "modules" else "")
    )
    ledger = _active_requirement_ledger(prompt)
'''
if field_depth_anchor not in text:
    raise SystemExit("field depth anchor missing")
text = text.replace(field_depth_anchor, field_depth_replacement, 1)

SOURCE.write_text(text, encoding="utf-8")

TEST.write_text(
    '''from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.spec import SpecValidationError


class _Router:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate_text(self, role, messages, **kwargs):
        del role, messages, kwargs
        return self.output


def _ledger():
    return (
        {
            "requirement_id": "req_space_mode_trading_b1d7cc479a",
            "capability": "space_mode_trading",
            "authored_text": "거래로 우주선 업그레이드를 구매한다.",
            "semantic_statement": "Trade resources for spaceship upgrades.",
            "observable_behavior": {},
            "acceptance": ["Trading changes inventory and currency atomically."],
        },
    )


def test_unknown_requirement_id_in_progression_falls_back_field_locally(monkeypatch):
    monkeypatch.setenv("MMM_PLANNER_TRACE", "0")
    monkeypatch.setenv("MMM_PLANNER_TRACE_CONSOLE", "0")
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: _ledger())
    output = """## progression
- Currency supports `req_space_trading_protocol_001` (Inferred for context).
## combat
### encounters
- Combat remains independent of trading.
## mod_context
### authority
- Mutable economy state is server-authoritative.
"""
    section = design._generate_section(
        _Router(output),
        prompt="space trading",
        section_id="systems_and_progression",
        fields=("progression", "combat", "mod_context"),
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert section["progression"] == ["Trade resources for spaceship upgrades."]
    assert section["combat"] == {"encounters": ["Combat remains independent of trading."]}
    assert section["mod_context"] == {"authority": ["Mutable economy state is server-authoritative."]}


def test_unknown_requirement_id_is_rejected_in_nested_map():
    with pytest.raises(SpecValidationError, match="unknown requirement ids"):
        design._validate_section_types(
            {"combat": {"encounter": ["Uses req_not_approved_123."]}},
            ("combat",),
            requirement_ids=("req_space_mode_trading_b1d7cc479a",),
        )


def test_exact_approved_requirement_id_is_accepted():
    design._validate_section_types(
        {"progression": ["Use `req_space_mode_trading_b1d7cc479a` for the trade loop."]},
        ("progression",),
        requirement_ids=("req_space_mode_trading_b1d7cc479a",),
    )


def test_non_module_stage_prompt_does_not_receive_modules_section_contract(monkeypatch):
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: _ledger())
    messages = design._section_messages(
        prompt="space trading",
        section_id="systems_and_progression",
        fields=("progression", "combat", "mod_context"),
        research={},
    )
    system = messages[0]["content"]
    assert "The modules section is the implementation-leaf index" not in system
    assert "MODULE LEAF INDEX" not in system
    assert "requirement_refs" not in system
    assert "Never invent requirement IDs" in system


def test_modules_stage_receives_leaf_index_contract(monkeypatch):
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: _ledger())
    messages = design._section_messages(
        prompt="space trading",
        section_id="modules_and_assets",
        fields=("modules", "assets"),
        research={},
    )
    system = messages[0]["content"]
    assert "MODULE LEAF INDEX" in system
    assert "requirement_refs" in system
''',
    encoding="utf-8",
)
