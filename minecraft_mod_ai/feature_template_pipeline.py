"""Fixed feature decomposition pipeline with host-owned atomicity decisions."""
from copy import deepcopy

from .task_template_runner import TemplateBlocked, run_record_template


FEATURE_DETAIL_STEPS = (
    "purpose", "behavior", "trigger", "input", "output", "state", "transition",
    "rules", "constraints", "dependencies", "connections", "persistence",
    "networking", "ui", "resources", "assets",
)

FEATURE_CONTEXT_DEPENDENCIES = {
    "purpose": (),
    "behavior": ("purpose",),
    "trigger": ("behavior",),
    "input": ("behavior", "trigger"),
    "output": ("behavior",),
    "state": ("behavior",),
    "transition": ("state", "trigger"),
    "rules": ("behavior",),
    "constraints": ("behavior",),
    "dependencies": ("behavior",),
    "connections": ("dependencies",),
    "persistence": ("state",),
    "networking": ("behavior", "state"),
    "ui": ("behavior", "input", "output"),
    "resources": ("behavior",),
    "assets": ("ui", "resources"),
}

ATOMIC_CHECKS = (
    "single_primary_behavior", "explicit_trigger", "explicit_inputs", "explicit_outputs",
    "state_known", "dependencies_known", "external_effects_known",
    "minecraft_mapping_possible", "required_artifacts_known", "test_case_writable",
)

ATOMIC_CONTEXT_DEPENDENCIES = {
    "single_primary_behavior": ("behavior",),
    "explicit_trigger": ("trigger",),
    "explicit_inputs": ("input",),
    "explicit_outputs": ("output",),
    "state_known": ("state", "transition"),
    "dependencies_known": ("dependencies",),
    "external_effects_known": ("output", "connections", "rules"),
    "minecraft_mapping_possible": ("behavior", "dependencies", "resources"),
    "required_artifacts_known": ("persistence", "networking", "ui", "resources", "assets"),
    "test_case_writable": ("behavior", "trigger", "input", "output", "rules"),
}


def evaluate_atomicity(check_records):
    """Return host-derived atomicity; model output never supplies this value."""
    by_name = {}
    for record in check_records:
        name = record.get("check")
        if name not in ATOMIC_CHECKS:
            raise ValueError(f"FEATURE_ATOMIC_CHECK: unknown check {name!r}")
        if name in by_name:
            raise ValueError(f"FEATURE_ATOMIC_CHECK: duplicate check {name}")
        if not isinstance(record.get("passed"), bool):
            raise ValueError(f"FEATURE_ATOMIC_CHECK: non-boolean result for {name}")
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"FEATURE_ATOMIC_CHECK: missing reason for {name}")
        by_name[name] = deepcopy(record)
    missing = [name for name in ATOMIC_CHECKS if name not in by_name]
    if missing:
        raise ValueError("FEATURE_ATOMIC_CHECK: missing checks: " + ", ".join(missing))
    failed = [name for name in ATOMIC_CHECKS if not by_name[name]["passed"]]
    return {"atomic": not failed, "failed_checks": failed,
            "checks": [by_name[name] for name in ATOMIC_CHECKS]}


def _run_atomic_checks(router, feature_record, *, allowed_refs, progress=None, checkpoint=None):
    records = []
    for check in ATOMIC_CHECKS:
        sections = feature_record["sections"]
        context = {
            "feature_id": feature_record["feature_id"],
            "feature_description": feature_record["feature_description"],
            "target_check": check,
            "relevant_sections": {name: deepcopy(sections[name]) for name in ATOMIC_CONTEXT_DEPENDENCIES[check]},
        }
        result = run_record_template(router, "feature/atomic_check", context=context,
            allowed_refs=allowed_refs, progress=progress, checkpoint=checkpoint)
        if result["reason"] or len(result["records"]) != 1:
            raise TemplateBlocked(f"FEATURE_ATOMIC_CHECK: {check} must produce exactly one check record")
        record = result["records"][0]
        if record["check"] != check:
            raise ValueError(f"FEATURE_ATOMIC_CHECK: expected {check}, got {record['check']}")
        records.append(record)
    return evaluate_atomicity(records)


def complete_feature(router, feature, *, allowed_refs, progress=None, checkpoint=None,
                     depth=0, max_depth=32, ancestry=()):
    """Complete one feature; recursively split it until every leaf passes all checks."""
    if depth > max_depth:
        raise TemplateBlocked("FEATURE_DECOMPOSE: maximum decomposition depth exceeded")
    feature_id = feature.get("feature_id")
    description = feature.get("feature_description") or feature.get("normalized_description")
    if not isinstance(feature_id, str) or not feature_id.strip():
        raise ValueError("FEATURE_INPUT: feature_id is required")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("FEATURE_INPUT: feature description is required")
    if feature_id in ancestry:
        raise TemplateBlocked(f"FEATURE_DECOMPOSE: ancestry cycle at {feature_id}")

    base = {"feature_id": feature_id, "feature_description": description}
    sections, evidence_refs = {}, []
    for step in FEATURE_DETAIL_STEPS:
        relevant = {name: deepcopy(sections[name]) for name in FEATURE_CONTEXT_DEPENDENCIES[step]}
        result = run_record_template(router, f"feature/{step}",
            context={"feature": base, "relevant_sections": relevant},
            allowed_refs=allowed_refs, progress=progress, checkpoint=checkpoint)
        sections[step] = {"records": result["records"], "not_applicable_reason": result["reason"]}
        for ref in result["evidence_refs"]:
            if ref not in evidence_refs:
                evidence_refs.append(ref)

    completed = {**base, "sections": sections}
    atomicity = _run_atomic_checks(router, completed, allowed_refs=allowed_refs,
        progress=progress, checkpoint=checkpoint)
    node = {**completed, "atomicity": atomicity, "children": [], "evidence_refs": evidence_refs}
    if atomicity["atomic"]:
        return node

    decomposition = run_record_template(router, "feature/decompose",
        context={"feature": completed, "failed_atomic_checks": atomicity["failed_checks"]},
        allowed_refs=allowed_refs, progress=progress, checkpoint=checkpoint)
    children = decomposition["records"]
    if not children:
        raise TemplateBlocked(f"FEATURE_DECOMPOSE: non-atomic feature {feature_id} produced no children")
    child_ids = [child["feature_id"] for child in children]
    if len(child_ids) != len(set(child_ids)):
        raise TemplateBlocked(f"FEATURE_DECOMPOSE: duplicate child id under {feature_id}")
    if feature_id in child_ids:
        raise TemplateBlocked(f"FEATURE_DECOMPOSE: child repeats parent {feature_id}")

    next_ancestry = ancestry + (feature_id,)
    for child in children:
        node["children"].append(complete_feature(router,
            {"feature_id": child["feature_id"], "feature_description": child["behavior"]},
            allowed_refs=allowed_refs, progress=progress, checkpoint=checkpoint,
            depth=depth + 1, max_depth=max_depth, ancestry=next_ancestry))
    return node


def atomic_leaves(feature_tree):
    """Yield only leaves accepted by the host-owned atomicity gate."""
    if feature_tree["atomicity"]["atomic"]:
        if feature_tree["children"]:
            raise ValueError("FEATURE_TREE: atomic node cannot have children")
        yield feature_tree
        return
    if not feature_tree["children"]:
        raise ValueError("FEATURE_TREE: non-atomic node must have children")
    for child in feature_tree["children"]:
        yield from atomic_leaves(child)
