"""Preserve request scope by authoring one bounded worksheet DAG per feature."""
from collections.abc import Mapping
from copy import deepcopy
import json

from . import authored_structured_design as design
from .planning_detail_slots import DETAIL_RECORDS
from .state_definition_merge import merge_state_definitions
from .request_record_ownership import bind_request_records


def _unique_rows(rows):
    result = {}
    for row in rows:
        identity = json.dumps(row, ensure_ascii=False, sort_keys=True)
        result.setdefault(identity, deepcopy(row))
    return list(result.values())


def _merge_section(section, prior, addition):
    specs = [row.get("specification", {}) for row in (prior, addition)]
    specification = {}
    for concern in DETAIL_RECORDS[section]:
        before, after = (spec.get(concern, []) for spec in specs)
        specification[concern] = (
            merge_state_definitions(before, after)
            if (section, concern) == ("state_model", "variables")
            else _unique_rows([*before, *after])
        )
    reasons = {
        row["concern"]: deepcopy(row)
        for spec in specs for row in spec.get("inapplicable_concerns", ())
        if not specification[row["concern"]]
    }
    specification["inapplicable_concerns"] = list(reasons.values())
    return {
        "specification": specification,
        "constraint_evidence_refs": list(dict.fromkeys(
            ref for row in (prior, addition)
            for ref in row.get("constraint_evidence_refs", ())
        )),
    }


def merge_request_sections(prior, addition):
    """Accumulate validated records, preserving shared state and source effects."""
    result = design.normalize_structured_sections(prior)
    for section, row in design.normalize_structured_sections(addition).items():
        result[section] = _merge_section(section, result.get(section, {}), row)
    return design.normalize_structured_sections(result)


def _feature_prompt(original, feature, shared_variables):
    return (
        "Original request (context; do not author neighboring features):\n" + original
        + "\n\nACTIVE REQUEST FEATURE:\n"
        + json.dumps({
            "requirement_id": feature["requirement_id"],
            "statement": feature["statement"],
            "source_text": feature.get("source_span", {}).get("text", ""),
        }, ensure_ascii=False, separators=(",", ":"))
        + "\nAuthor this feature's concrete activation, result and implementation only. "
        "Keep its state changes, UI interactions and verification in the same scope. "
        "Preserve every distinct action within this feature. Opening a screen alone "
        "does not implement the requested gameplay."
        + "\n\nPreviously accepted shared state declarations (read-only):\n"
        + json.dumps(shared_variables, ensure_ascii=False, separators=(",", ":"))
        + "\nThese symbols are available to state expressions and mutations. Declare "
        "only additional state, or repeat a shared declaration exactly. Do not change "
        "its type, default, owner, unit or domain to fit this feature."
    )


def author_request_structured_sections(
    router, prompt, request_scope, *, media_paths=(), budget=None,
    checkpoint=None,
):
    """The host iterates source requirements; the model cannot omit a neighbor."""
    completed = {}
    requirements = request_scope.get("requirements", ())
    if not requirements:
        return design.author_structured_sections(
            router, prompt, media_paths=media_paths, budget=budget,
        )
    original = str(request_scope.get("requested_prompt") or prompt)
    for feature in requirements:
        if not isinstance(feature, Mapping):
            raise ValueError("REQUEST_DESIGN_FEATURE_OBJECT_REQUIRED")
        variables = (
            completed.get("state_model", {}).get("specification", {})
            .get("variables", [])
        )
        sections = checkpoint.load_request_design(feature, variables) if checkpoint else None
        if sections is not None:
            try:
                sections = design.normalize_structured_sections(sections)
            except ValueError:
                sections = None
        if sections is None:
            focused = _feature_prompt(original, feature, variables)
            sections = design.author_structured_sections(
                router, focused, media_paths=media_paths, budget=budget,
                shared_state_variables=variables,
            )
        sections = bind_request_records(sections, feature["requirement_id"])
        completed = merge_request_sections(completed, sections)
        if checkpoint:
            checkpoint.save_request_design(feature, variables, sections)
        print("REQUEST_DESIGN: accepted " + str(feature["requirement_id"]), flush=True)
    return completed


def _unbound_request_actions(sections):
    """Find author-owned server operations without a same-trigger state effect.

    UI presentation is not a transaction. Namesake triggers from other
    requirements are never acceptable evidence of ownership.
    """
    from .authored_ui_action_bindings import is_presentation_only_interaction
    from .request_record_ownership import SOURCE_REQUIREMENT, same_action_source

    specs = design.normalize_structured_sections(sections)
    state = specs.get("state_model", {}).get("specification", {})
    interactions = specs.get("resources_and_ui", {}).get(
        "specification", {}).get("interactions", ())
    effects = [
        row for concern in ("updates", "transitions")
        for row in state.get(concern, ())
        if isinstance(row, Mapping) and row.get("mutation")
    ]
    missing = []
    for row in interactions:
        action = row.get("server_action")
        if not isinstance(action, (str, Mapping)) or not action:
            continue
        if is_presentation_only_interaction(row, state):
            continue
        if any(same_action_source(effect, row) for effect in effects):
            continue
        requirement = row.get(SOURCE_REQUIREMENT)
        if not isinstance(requirement, str) or not requirement:
            raise ValueError(
                "ACTION_REPAIR_SOURCE_UNKNOWN: non-presentation action has "
                "no original request owner"
            )
        missing.append({
            "source_requirement_id": requirement,
            "trigger": row.get("trigger", ""),
            "server_action": deepcopy(action),
        })
    return missing


def repair_request_action_effects(
    router, sections, request_scope, *, requested_prompt="", budget=None,
    media_paths=(),
):
    """One bounded, source-preserving semantic repair before code generation.

    The host never supplies or guesses a mutation. It asks the planner to
    reauthor the *state_model* for each deficient request feature while
    keeping all existing UI commands and unrelated request records intact.
    If the result still cannot execute, reject it rather than forging success.
    """
    from .request_record_ownership import SOURCE_REQUIREMENT, bind_request_records
    from .typed_action_authoring import (
        preflight_state_guard_semantics, preflight_ui_action_effects,
    )

    original = design.normalize_structured_sections(sections)
    missing = _unbound_request_actions(original)
    if not missing:
        preflight_state_guard_semantics(original)
        preflight_ui_action_effects(original, strict_source=True)
        return original
    features = {
        row.get("requirement_id"): row
        for row in request_scope.get("requirements", ())
        if isinstance(row, Mapping)
    }
    unresolved = sorted({row["source_requirement_id"] for row in missing})
    for requirement_id in unresolved:
        if requirement_id not in features:
            raise ValueError(
                "ACTION_REPAIR_REQUIREMENT_MISSING: " + requirement_id
            )

    # Repair one command at a time so a complex request is not limited to
    # four aggregate state rows. Do not erase already-valid same-feature rules.
    current = deepcopy(original)
    for requirement_id in unresolved:
        feature = features[requirement_id]
        subset = [row for row in missing
                  if row["source_requirement_id"] == requirement_id]
        triggers = [str(row["trigger"]).strip().casefold() for row in subset]
        if len(triggers) != len(set(triggers)):
            raise ValueError(
                "ACTION_REPAIR_AMBIGUOUS_TRIGGER: multiple actions share "
                "one source trigger within " + requirement_id
            )
        for action in subset:
            current_state = current.get("state_model", {}).get("specification", {})
            shared = deepcopy(current_state.get("variables", []))
            focused = _feature_prompt(
                str(request_scope.get("requested_prompt") or requested_prompt),
                feature, shared,
            )
            focused += (
                "\n\nMANDATORY EXECUTION CONTRACT REPAIR (one attempt):\n"
                "This ONE approved server command lacks a state effect. "
                "Do not rename or omit its original trigger or introduce a "
                "different UI command. Author a canonical state_model.update "
                "or transition using exactly this trigger, with real "
                "server-owned, guarded atomic mutations and any necessary "
                "state variable declarations. This is not a screen-open "
                "operation. Do not author another requirement's transaction "
                "or fabricate zero-valued effects. Target command:\n"
                + json.dumps(action, ensure_ascii=False, sort_keys=True)
                + "\nExisting canonical request state:\n"
                + json.dumps(current_state, ensure_ascii=False, sort_keys=True)
            )
            print(
                "ACTION_SOURCE_REPAIR: request=" + requirement_id
                + " trigger=" + str(action["trigger"])[:96], flush=True,
            )
            behavior = current.get("behavior_contract")
            if not isinstance(behavior, Mapping):
                raise ValueError(
                    "ACTION_SOURCE_REPAIR_BEHAVIOR_MISSING: " + requirement_id
                )
            candidate = design.author_state_repair_section(
                router, focused, behavior_contract=behavior,
                required_action_count=1,
                shared_state_variables=shared, media_paths=media_paths,
                budget=budget,
            )
            candidate = bind_request_records(candidate, requirement_id)
            candidate_state = candidate.get("state_model", {}).get(
                "specification", {})
            selected = {
                concern: [
                    deepcopy(row) for row in candidate_state.get(concern, ())
                    if str(row.get("trigger") or "").strip().casefold()
                    == str(action["trigger"]).strip().casefold()
                ]
                for concern in ("transitions", "updates")
            }
            if not any(selected.values()):
                raise ValueError(
                    "ACTION_SOURCE_REPAIR_EXHAUSTED: planner did not produce "
                    "an exact same-trigger state rule for " + requirement_id
                    + " / " + str(action["trigger"])[:128]
                )
            repaired = deepcopy(current)
            if "state_model" not in repaired:
                repaired["state_model"] = {
                    "constraint_evidence_refs": [], "specification": {
                        **{name: [] for name in DETAIL_RECORDS["state_model"]},
                        "inapplicable_concerns": [],
                    },
                }
            target = repaired["state_model"]["specification"]
            target["variables"] = merge_state_definitions(
                target.get("variables", []), candidate_state.get("variables", []),
            )
            for concern in ("transitions", "updates"):
                target[concern].extend(selected[concern])
            current = design.normalize_structured_sections(repaired)

    try:
        preflight_state_guard_semantics(current)
        preflight_ui_action_effects(current, strict_source=True)
    except ValueError as error:
        raise ValueError(
            "ACTION_SOURCE_REPAIR_EXHAUSTED: the model failed to author "
            "the source-owned state effects after one attempt per "
            "requirement; no unrelated state writes accepted. "
            + str(error)
        ) from error
    print(
        "ACTION_SOURCE_REPAIR: validated requirements="
        + ",".join(unresolved), flush=True,
    )
    return current
