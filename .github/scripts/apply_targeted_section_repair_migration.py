from __future__ import annotations

from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected migration anchor not found in {path}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_exact(
    "minecraft_mod_ai/planning_state_adaptive_implementation.py",
    "from .planning_detail_slots import DETAIL_RECORDS\n",
    "from .planning_detail_slots import DETAIL_RECORDS\n"
    "from .planning_targeted_section_repair import generate_targeted_section_fragment\n",
)

replace_exact(
    "minecraft_mod_ai/planning_state_adaptive_implementation.py",
    '''                supplement = generate_criterion_fragment(
                    router,
                    requirement=job["requirement"],
                    criterion=(
                        "Resolve the missing required worksheet section " + section
                        + " across these acceptance criteria: " + "; ".join(job["criteria"])
                        + ". Return a concrete implementation or constraint for this section. "
                        "For reuse, assess the supplied candidates and distinguish verified reuse, "
                        "reference-only evidence and missing proof. For verification, specify "
                        "observable success and rejection checks."
                    ),
                    selected_sections=(section,),
                    evidence=job["evidence"],
                    allowed_refs=job["allowed"],
                )
''',
    '''                supplement = generate_targeted_section_fragment(
                    router,
                    requirement=job["requirement"],
                    criterion=(
                        "Resolve the missing required worksheet section " + section
                        + " across these acceptance criteria: " + "; ".join(job["criteria"])
                        + ". Return a concrete implementation or constraint for this section. "
                        "For reuse, assess the supplied candidates and distinguish verified reuse, "
                        "reference-only evidence and missing proof. For verification, specify "
                        "observable success and rejection checks."
                    ),
                    selected_sections=job["selected_sections"],
                    target_section=section,
                    evidence=job["evidence"],
                    allowed_refs=job["allowed"],
                )
''',
)

replace_exact(
    "tests/test_planning_missing_section_targeted_repair.py",
    "import minecraft_mod_ai.planning_state_adaptive_implementation as adaptive\n",
    "import minecraft_mod_ai.planning_state_adaptive_implementation as adaptive\n"
    "from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS\n",
)

replace_exact(
    "tests/test_planning_missing_section_targeted_repair.py",
    '''        selected_sections: tuple[str, ...],
        evidence: list[dict[str, Any]],
        allowed_refs: set[str],
    ) -> dict[str, Any]:
        del requirement, criterion, evidence, allowed_refs
        selection = tuple(selected_sections)
        repair_selections.append(selection)
        assert len(selection) == 1
        section = selection[0]
        assert section in implementations
''',
    '''        selected_sections: tuple[str, ...],
        target_section: str,
        evidence: list[dict[str, Any]],
        allowed_refs: set[str],
    ) -> dict[str, Any]:
        del requirement, criterion, evidence, allowed_refs
        assert tuple(selected_sections) == WORKSHEET_SECTIONS
        section = target_section
        repair_selections.append((section,))
        assert section in implementations
''',
)

replace_exact(
    "tests/test_planning_missing_section_targeted_repair.py",
    'monkeypatch.setattr(adaptive, "generate_criterion_fragment", generate_targeted_fragment)',
    'monkeypatch.setattr(adaptive, "generate_targeted_section_fragment", generate_targeted_fragment)',
)

replace_exact(
    "tests/test_planning_missing_section_targeted_repair.py",
    '''        "selected_sections": (
            "behavior_contract",
            "integration",
            "persistence",
            "reuse_assessment",
        ),
''',
    '''        "selected_sections": WORKSHEET_SECTIONS,
''',
)

replace_exact(
    "tests/test_planning_missing_section_targeted_repair.py",
    '''                "section_updates": [
                    {
                        "section": "behavior_contract",
                        "implementation": "A successful collection exposes the acquired resource.",
                        "constraint": "Rejected collection leaves inventory unchanged.",
                        "evidence_refs": [],
                    }
                ]
''',
    '''                "section_updates": [
                    {
                        "section": section,
                        "implementation": f"Existing concrete contract for {section}.",
                        "constraint": f"Existing concrete boundary for {section}.",
                        "evidence_refs": [],
                    }
                    for section in WORKSHEET_SECTIONS
                    if section not in {"integration", "persistence", "reuse_assessment"}
                ]
''',
)

replace_exact(
    "tests/test_planning_detail_grounding.py",
    '    assert all(call["tool_name"] == "submit_fixed_template" for call in router.calls)\n',
    '    assert all(str(call["tool_name"]).startswith("submit_") for call in router.calls)\n'
    '    assert all(call["parameters"].get("type") == "object" for call in router.calls)\n',
)

print("targeted section repair migration applied")
