"""Fixed feature decomposition pipeline with host-owned atomicity and convergence."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .bounded_record_template import run_record_template
from .parallel_model_tasks import deterministic_model_map, serialized_callback
from .task_template_runner import TemplateBlocked


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


def _semantic_text(value):
    return " ".join(str(value or "").split()).casefold()


def _feature_signature(feature):
    """Identify semantic recurrence independently of generated feature IDs."""
    description = feature.get("feature_description") or feature.get("normalized_description")
    payload = {"description": _semantic_text(description)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


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
    return {
        "atomic": not failed,
        "failed_checks": failed,
        "checks": [by_name[name] for name in ATOMIC_CHECKS],
    }


def _run_atomic_checks(router, feature_record, *, allowed_refs, progress=None, checkpoint=None):
    sections = feature_record["sections"]
    safe_checkpoint = serialized_callback(checkpoint)

    def run_check(check):
        context = {
            "feature_id": feature_record["feature_id"],
            "feature_description": feature_record["feature_description"],
            "target_check": check,
            "relevant_sections": {
                name: deepcopy(sections[name])
                for name in ATOMIC_CONTEXT_DEPENDENCIES[check]
            },
        }
        result = run_record_template(
            router,
            "feature/atomic_check",
            context=context,
            allowed_refs=allowed_refs,
            progress=progress,
            checkpoint=safe_checkpoint,
        )
        if result["reason"] or len(result["records"]) != 1:
            raise TemplateBlocked(
                f"FEATURE_ATOMIC_CHECK: {check} must produce exactly one check record"
            )
        record = result["records"][0]
        if record["check"] != check:
            raise ValueError(f"FEATURE_ATOMIC_CHECK: expected {check}, got {record['check']}")
        return record

    records = deterministic_model_map(
        router,
        ATOMIC_CHECKS,
        run_check,
        role="planner",
        thread_name_prefix="feature-atomic-check",
    )
    return evaluate_atomicity(records)


def _run_feature_sections(router, base, *, allowed_refs, progress=None, checkpoint=None):
    """Execute the declared feature dependency DAG in deterministic ready waves."""

    sections = {}
    evidence_refs = []
    remaining = set(FEATURE_DETAIL_STEPS)
    safe_checkpoint = serialized_callback(checkpoint)

    while remaining:
        ready = tuple(
            step
            for step in FEATURE_DETAIL_STEPS
            if step in remaining
            and all(dependency in sections for dependency in FEATURE_CONTEXT_DEPENDENCIES[step])
        )
        if not ready:
            raise ValueError(
                "FEATURE_DETAIL_DAG: unresolved dependency cycle: "
                + ", ".join(step for step in FEATURE_DETAIL_STEPS if step in remaining)
            )

        section_snapshot = deepcopy(sections)

        def run_step(step):
            relevant = {
                name: deepcopy(section_snapshot[name])
                for name in FEATURE_CONTEXT_DEPENDENCIES[step]
            }
            return run_record_template(
                router,
                f"feature/{step}",
                context={"feature": base, "relevant_sections": relevant},
                allowed_refs=allowed_refs,
                progress=progress,
                checkpoint=safe_checkpoint,
            )

        results = deterministic_model_map(
            router,
            ready,
            run_step,
            role="planner",
            thread_name_prefix="feature-detail",
        )
        for step, result in zip(ready, results):
            sections[step] = {
                "records": result["records"],
                "not_applicable_reason": result["reason"],
            }
            for ref in result["evidence_refs"]:
                if ref not in evidence_refs:
                    evidence_refs.append(ref)
            remaining.remove(step)

    return sections, evidence_refs


def complete_feature(
    router,
    feature,
    *,
    allowed_refs,
    progress=None,
    checkpoint=None,
    ancestry=(),
    ancestry_signatures=(),
    parent_failed_checks=None,
):
    """Recursively split while the finite unresolved atomicity set strictly decreases."""

    feature_id = feature.get("feature_id")
    description = feature.get("feature_description") or feature.get("normalized_description")
    if not isinstance(feature_id, str) or not feature_id.strip():
        raise ValueError("FEATURE_INPUT: feature_id is required")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("FEATURE_INPUT: feature description is required")
    if feature_id in ancestry:
        raise TemplateBlocked(f"FEATURE_DECOMPOSE: ancestry id cycle at {feature_id}")

    signature = _feature_signature(feature)
    if signature in ancestry_signatures:
        raise TemplateBlocked(
            f"FEATURE_DECOMPOSE: semantic ancestry cycle/no-progress at {feature_id}"
        )

    base = {"feature_id": feature_id, "feature_description": description}
    sections, evidence_refs = _run_feature_sections(
        router,
        base,
        allowed_refs=allowed_refs,
        progress=progress,
        checkpoint=checkpoint,
    )

    completed = {**base, "sections": sections}
    atomicity = _run_atomic_checks(
        router,
        completed,
        allowed_refs=allowed_refs,
        progress=progress,
        checkpoint=checkpoint,
    )
    current_failed = frozenset(atomicity["failed_checks"])
    if parent_failed_checks is not None:
        previous_failed = frozenset(parent_failed_checks)
        if current_failed and not current_failed < previous_failed:
            raise TemplateBlocked(
                "FEATURE_DECOMPOSE: unresolved atomic checks did not strictly decrease"
            )
    node = {
        **completed,
        "atomicity": atomicity,
        "children": [],
        "evidence_refs": evidence_refs,
    }
    if atomicity["atomic"]:
        return node

    decomposition = run_record_template(
        router,
        "feature/decompose",
        context={
            "feature": completed,
            "failed_atomic_checks": atomicity["failed_checks"],
        },
        allowed_refs=allowed_refs,
        progress=progress,
        checkpoint=checkpoint,
    )
    children = decomposition["records"]
    if not children:
        raise TemplateBlocked(
            f"FEATURE_DECOMPOSE: non-atomic feature {feature_id} produced no children"
        )

    child_ids = [child["feature_id"] for child in children]
    if len(child_ids) != len(set(child_ids)):
        raise TemplateBlocked(f"FEATURE_DECOMPOSE: duplicate child id under {feature_id}")
    if feature_id in child_ids:
        raise TemplateBlocked(f"FEATURE_DECOMPOSE: child repeats parent {feature_id}")

    child_features = [
        {"feature_id": child["feature_id"], "feature_description": child["behavior"]}
        for child in children
    ]
    child_signatures = [_feature_signature(child) for child in child_features]
    if len(child_signatures) != len(set(child_signatures)):
        raise TemplateBlocked(
            f"FEATURE_DECOMPOSE: semantically duplicate children under {feature_id}"
        )
    if signature in child_signatures:
        raise TemplateBlocked(
            f"FEATURE_DECOMPOSE: decomposition made no semantic progress for {feature_id}"
        )
    inherited = set(ancestry_signatures)
    inherited.add(signature)
    repeated = [sig for sig in child_signatures if sig in inherited]
    if repeated:
        raise TemplateBlocked(
            f"FEATURE_DECOMPOSE: child returns to an ancestor under {feature_id}"
        )

    next_ancestry = ancestry + (feature_id,)
    next_signatures = ancestry_signatures + (signature,)
    for child in child_features:
        node["children"].append(
            complete_feature(
                router,
                child,
                allowed_refs=allowed_refs,
                progress=progress,
                checkpoint=checkpoint,
                ancestry=next_ancestry,
                ancestry_signatures=next_signatures,
                parent_failed_checks=atomicity["failed_checks"],
            )
        )
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


def discover_features(router, *, context, allowed_refs=(), progress=None, checkpoint=None):
    result = run_record_template(
        router,
        "feature/discover",
        context=context,
        allowed_refs=allowed_refs,
        progress=progress,
        checkpoint=checkpoint,
    )
    return result["records"]


def decompose_features_pipeline(
    router,
    features,
    *,
    allowed_refs=(),
    progress=None,
    checkpoint=None,
):
    atomic_results = []
    for feature in features:
        tree = complete_feature(
            router,
            feature,
            allowed_refs=allowed_refs,
            progress=progress,
            checkpoint=checkpoint,
        )
        atomic_results.extend(atomic_leaves(tree))
    return atomic_results
