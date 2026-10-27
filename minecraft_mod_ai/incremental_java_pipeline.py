"""Production integration for incremental source programs and per-operation repairs."""
from __future__ import annotations

from itertools import count
import json


def compose_source(router, spec, messages, *, source_context, construction_context, behaviors, legacy_author):
    if construction_context.get("host_direct_feature_source"):
        from .incremental_java_author import compose_incremental_feature
        source, program = compose_incremental_feature(router, spec, messages,
            construction_context=construction_context, behaviors=behaviors)
        construction_context["incremental_state"]["program"] = program
        return source
    return legacy_author(router, spec, messages, source_context=source_context,
        construction_context=construction_context, behaviors=behaviors)


def candidate_data(source, diagnostic, state):
    result = {"source": source, "diagnostic": diagnostic}
    if state.get("program"):
        result["program_json"] = json.dumps(state["program"], ensure_ascii=False, separators=(",", ":"))
    return result


def load_candidate(pending, state, direct):
    if not pending:
        return pending
    if set(pending) - {"source", "diagnostic", "program_json"} or not {"source", "diagnostic"} <= set(pending):
        raise ValueError("JAVA_HOST_CANDIDATE_INVALID")
    if direct:
        if not isinstance(pending.get("program_json"), str):
            # Legacy whole-body candidates are not a new protocol's accepted AST.
            return {}
        state["program"] = json.loads(pending["program_json"])
    return pending


def compile_attempts(direct, legacy_budget):
    # Independent operation errors do not consume one shared whole-feature retry.
    # Cancellation/deadlines still bound model execution. Repeated failures of
    # the same cell are counted in repair_member, without discarding other work.
    return count() if direct else range(legacy_budget + 1)


def exhausted(direct, attempt, legacy_budget):
    return not direct and attempt == legacy_budget


def repair_member(router, source, diagnostic, *, context):
    from .incremental_java_repair import repair_incremental_program
    state = context["incremental_state"]
    source, program = repair_incremental_program(router, source, diagnostic,
        program=state["program"], spec=context["incremental_spec"],
        messages=context["incremental_messages"], construction_context=context,
        behaviors=context["behaviors"])
    state["program"] = program
    return source


def verify_full_source(source, spec, context, *, package, name, source_context=(), verifier, state_validators):
    from .incremental_java_state import validate_program_state, map_behavior_diagnostic
    validate_program_state(source, spec, context, validators=state_validators)
    try:
        return verifier(source, spec, context, package=package, name=name, source_context=source_context)
    except ValueError as exc:
        diagnostic = map_behavior_diagnostic(source, spec, context, str(exc))
        if diagnostic != str(exc):
            raise ValueError(diagnostic) from exc
        raise
