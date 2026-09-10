"""Repair a host-selected section using its individual concern templates."""
from .planning_detail_template import normalize_required_sections
from .planning_criterion_fragments import generate_section_records


def generate_targeted_section_fragment(router, *, requirement, criterion, selected_sections, target_section, evidence, allowed_refs, progress=None, checkpoint=None):
    if target_section not in normalize_required_sections(selected_sections):
        raise ValueError("DETAILED_PLAN_TARGET_SECTION: target is outside selected sections")
    return {"section_updates": [generate_section_records(router, requirement=requirement,
        criterion=criterion, section=target_section, evidence=evidence, allowed_refs=allowed_refs,
        progress=progress, checkpoint=checkpoint)]}
