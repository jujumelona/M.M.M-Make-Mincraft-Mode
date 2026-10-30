"""Production integration for incremental source programs and per-operation repairs."""
from __future__ import annotations

from itertools import count
import json


def _compose_source_owned(router, spec, messages, *,
                          source_context, construction_context, behaviors, legacy_author):
    from .host_source_strategy import prepare_source_owned_strategy

    state = construction_context["incremental_state"]
    state.pop("program", None)
    state["strategy"] = "host_source_owned"
    host_context = prepare_source_owned_strategy(
        construction_context, behaviors,
        side=str(spec.get("side") or "COMMON").upper(),
    )
    return legacy_author(
        router, spec, messages, source_context=source_context,
        construction_context=host_context, behaviors=behaviors,
    )


def compose_source(router, spec, messages, *, source_context, construction_context, behaviors, legacy_author):
    if construction_context.get("host_direct_feature_source"):
        from .host_source_strategy import prefer_source_owned_strategy
        if prefer_source_owned_strategy(router, construction_context, behaviors):
            return _compose_source_owned(
                router, spec, messages, source_context=source_context,
                construction_context=construction_context, behaviors=behaviors,
                legacy_author=legacy_author,
            )
        from .incremental_java_author import compose_incremental_feature
        from .source_occurrence_error import SourceHookUnavailable
        try:
            source, program = compose_incremental_feature(router, spec, messages,
                construction_context=construction_context, behaviors=behaviors)
        except (SourceHookUnavailable, RuntimeError) as exc:
            # Switch to a different *implementation engine* only for a missing
            # original occurrence or the proven no-progress terminal. Compiler
            # exceptions and unrelated model/runtime failures are not a
            # license to conceal errors behind another authoring strategy.
            if not isinstance(exc, SourceHookUnavailable) and not str(exc).startswith(
                ("INCREMENTAL_JAVA_NO_SEMANTIC_PROGRESS:",
                 "INCREMENTAL_JAVA_NO_AVAILABLE_OPERATION:")
            ):
                raise
            # The incremental registrar cannot invent an original occurrence.
            # Escalate the *strategy*, not the prompt or the 9B model's retry
            # counter. The host-first source author can construct a real
            # project-owned player action or a checked source-site Mixin and
            # binds the resulting exact callback ABI before authoring effects.
            # It still rejects missing source sites and unverified gameplay.
            return _compose_source_owned(
                router, spec, messages, source_context=source_context,
                construction_context=construction_context, behaviors=behaviors,
                legacy_author=legacy_author,
            )
        construction_context["incremental_state"]["program"] = program
        construction_context["incremental_state"]["strategy"] = "incremental_java"
        return source
    return legacy_author(router, spec, messages, source_context=source_context,
        construction_context=construction_context, behaviors=behaviors)


def candidate_data(source, diagnostic, state):
    result = {"source": source, "diagnostic": diagnostic}
    strategy = state.get("strategy")
    if strategy in {"incremental_java", "host_source_owned"}:
        result["strategy"] = strategy
    if strategy != "host_source_owned" and state.get("program"):
        result["program_json"] = json.dumps(state["program"], ensure_ascii=False, separators=(",", ":"))
    if strategy == "host_source_owned":
        receipts = state.get("native_mutation_receipts") or []
        result["native_mutation_receipts_json"] = json.dumps(
            receipts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return result


def load_candidate(pending, state, direct):
    if not pending:
        return pending
    allowed = {"source", "diagnostic", "program_json", "strategy", "native_mutation_receipts_json"}
    if set(pending) - allowed or not {"source", "diagnostic"} <= set(pending):
        raise ValueError("JAVA_HOST_CANDIDATE_INVALID")
    strategy = pending.get("strategy")
    if strategy not in {None, "incremental_java", "host_source_owned"}:
        raise ValueError("JAVA_HOST_CANDIDATE_STRATEGY_INVALID")
    if strategy == "host_source_owned":
        if not direct or "program_json" in pending:
            raise ValueError("JAVA_HOST_SOURCE_OWNERSHIP_CANDIDATE_INVALID")
        raw = pending.get("native_mutation_receipts_json")
        if not isinstance(raw, str):
            raise ValueError("JAVA_HOST_SOURCE_NATIVE_PROOF_MISSING")
        receipts = json.loads(raw)
        if not isinstance(receipts, list) or not all(isinstance(x, dict) for x in receipts):
            raise ValueError("JAVA_HOST_SOURCE_NATIVE_PROOF_INVALID")
        state.pop("program", None)
        state["native_mutation_receipts"] = receipts
        state["strategy"] = strategy
        return pending
    if direct:
        if not isinstance(pending.get("program_json"), str):
            # Legacy whole-body candidates are not a new protocol's accepted AST.
            return {}
        state["program"] = json.loads(pending["program_json"])
        state["strategy"] = "incremental_java"
    return pending


def has_incremental_ast(direct_feature, state):
    """Only a real incremental AST may enter the unbounded local repair loop.

    Host-generated entities, registry holders, rule templates and source-owned
    files are all non-AST products even when requested from the same Fabric
    feature entrypoint. Their diagnostics belong to bounded source repair.
    """
    if not direct_feature or state.get("strategy") != "incremental_java":
        return False
    if not isinstance(state.get("program"), dict):
        raise ValueError("JAVA_HOST_INCREMENTAL_AST_MISSING")
    return True


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


def _verify_observable_program(context):
    program = (context.get("incremental_state") or {}).get("program")
    if program:
        from .incremental_java_author import _assert_live_behavior_entrypoints
        _assert_live_behavior_entrypoints(program)


def verify_full_source(source, spec, context, *, package, name, source_context=(), verifier, state_validators):
    from .incremental_java_state import validate_program_state, map_behavior_diagnostic
    validate_program_state(source, spec, context, validators=state_validators)
    try:
        verified = verifier(source, spec, context, package=package, name=name, source_context=source_context)
        _verify_observable_program(context)
        return verified
    except (ValueError, RuntimeError) as exc:
        diagnostic = map_behavior_diagnostic(source, spec, context, str(exc))
        if diagnostic != str(exc):
            raise ValueError(diagnostic) from exc
        raise
