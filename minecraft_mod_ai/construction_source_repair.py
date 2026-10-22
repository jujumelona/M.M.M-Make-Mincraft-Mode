"""Preserve failed Java bytes and isolate direct-source recovery decisions."""
from __future__ import annotations

import json
from collections.abc import Mapping


def assemble_java_implementation(response: str, frame: Mapping) -> str:
    """Decode implementation data once; only the host writes declaration syntax."""
    from .model_adapters.base import ModelConfigurationError

    try:
        value = json.loads(response)
    except json.JSONDecodeError as cause:
        # Open-source json-repair handles *transport syntax* only. Do not use it
        # when the envelope was valid JSON but the Java string is invalid:
        # changing Java expressions silently would corrupt feature semantics.
        try:
            from json_repair import loads as repair_json_loads
            value = repair_json_loads(response, skip_json_loads=True)
        except (ValueError, TypeError, ImportError) as repair_error:
            error = ModelConfigurationError(
                "CONSTRUCTION_SOURCE_IMPLEMENTATION_INVALID: " + str(repair_error))
            error.failed_response = response
            raise error from cause
    if (not isinstance(value, dict) or set(value) != {"implementation"}
            or not isinstance(value["implementation"], str)):
        error = ModelConfigurationError(
            "CONSTRUCTION_SOURCE_IMPLEMENTATION_INVALID: expected exactly one "
            "implementation string")
        error.failed_response = response
        raise error
    body = value["implementation"]
    source = (frame["empty_value"] if frame.get("kind") == "field"
              and body == "" and "empty_value" in frame
              else frame["prefix"] + body + frame["suffix"])
    try:
        from .first_pass_java_declarations import validate_source_frame
        validate_source_frame(source, frame)
    except ValueError as error:
        # The host owns the member braces and the JSON transport envelope.
        # Some source coders copy its last delimiter into the decoded Java
        # string (e.g. 'return PASS;"}'). Reclaim ONLY an exact delimiter
        # suffix, and ONLY if the immutable Java frame parses afterward.
        # This is not an arbitrary string cleanup or an escape hatch for
        # broken Java statements: both original and recovered responses are
        # preserved for diagnostic-guided retries.
        if frame.get("kind") in {"method", "constructor"}:
            trimmed = body.rstrip()
            for transport_suffix in ('"}', '}'):
                if not trimmed.endswith(transport_suffix):
                    continue
                candidate_body = trimmed[:-len(transport_suffix)].rstrip()
                candidate = frame["prefix"] + candidate_body + frame["suffix"]
                try:
                    validate_source_frame(candidate, frame)
                except ValueError:
                    continue
                source = candidate
                error = None
                break
        if error is not None:
            error.failed_source = source
            error.failed_response = response
            raise
    # Syntax normalization and semantic admission are separate stages. A
    # corrected wrapper must expose its real state/API error to the retry;
    # otherwise the model sees the obsolete brace error and repeats its body.
    try:
        return validate_generated_java_source(
            source, frame, frame["prefix"], frame["suffix"])
    except ValueError as error:
        error.failed_response = response
        raise


def repeated_invalid_java_implementation(first: str, second: str) -> bool:
    """Detect a repeated rejected Java body, ignoring envelope-only whitespace.

    This does not rewrite or normalize executable Java. Two attempts that
    repeat exactly the same statements (modulo outer blank lines) provide no
    new implementation and must never be reported as a repair.
    """
    try:
        left, right = json.loads(first), json.loads(second)
    except (TypeError, ValueError):
        return first.strip() == second.strip()
    if (isinstance(left, dict) and isinstance(right, dict)
            and isinstance(left.get("implementation"), str)
            and isinstance(right.get("implementation"), str)):
        return left["implementation"].strip() == right["implementation"].strip()
    return first.strip() == second.strip()


def reject_repeated_invalid_java_implementation(first: str, second: str,
                                                error: ValueError) -> None:
    """Preserve the original parser or transport exception and its diagnostics.

    A repeated invalid Java reply is not a new revision. Never replace the
    JavaRegionParseError category with a generic configuration exception.
    """
    if not repeated_invalid_java_implementation(first, second):
        return
    error.failed_response = second
    error.repeated_invalid_response = True
    print("CONSTRUCTION_JAVA_REPEATED_REJECTED_BODY: "
          + str(error)[:1000], flush=True)
    raise error


def validate_generated_java_source(result, frame: Mapping, prefix: str, suffix: str) -> str:
    from .first_pass_java_declarations import validate_source_frame
    from .model_adapters.base import ModelConfigurationError

    if (not isinstance(result, str) or not (
            result == frame.get("empty_value")
            or (result.startswith(prefix) and result.endswith(suffix)))):
        error = ModelConfigurationError("CONSTRUCTION_SOURCE_SIGNATURE_INVALID")
        error.failed_source = result
        raise error
    try:
        validate_source_frame(result, frame)
        from .host_typed_state_access import (
            validate_host_state_access, validate_behavior_state_provenance,
        )
        contracts = frame.get("host_state_access", ())
        validate_host_state_access(result, contracts)
        validate_behavior_state_provenance(
            result, contracts, frame.get("host_behavior_contract"),
        )
    except ValueError as error:
        error.failed_source = result
        raise
    return result



def focused_java_correction_messages(
    messages, *, frame: Mapping, response: str, diagnostic: str,
) -> list[dict[str, str]]:
    """A failed member gets one NEW scoped correction, not its 50K prompt replay.

    Preserve the actual member work order, immutable signature, state access
    and selected inspected API. Irrelevant owner catalogs and completed class
    source are not reintroduced. The original evidence and generated bytes are
    still subject to the unchanged compiler gates.
    """
    selected = {}
    context = {}
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(message.get("content") or "")
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if "response_source_frame" in payload:
            selected = payload
        if isinstance(payload.get("inspected_target_api"), dict):
            from .construction_file_api_evidence import scoped_body_api_records
            evidence = payload["inspected_target_api"]
            context["selected_target_api"] = evidence.get("target_api", ())
            context["typed_input_producers"] = evidence.get("typed_input_producers", ())
            context["class_members"] = scoped_body_api_records(evidence)
        if "materialized_prerequisite_api" in payload:
            context["materialized_prerequisite_api"] = payload["materialized_prerequisite_api"]
        if "host_state_access" in payload:
            context["host_state_access"] = payload["host_state_access"]
    if not selected:
        # Non-host callers may rely on a free-form last user instruction.
        selected = {"original_request": next((
            message.get("content", "") for message in reversed(messages)
            if message.get("role") == "user"
        ), "")}
    selected_fields = (
        "requirement", "current_unit_purpose", "member_role", "assigned_behavior",
        "host_signature", "declared_member_api", "declaration_metadata",
        "verified_event_callback", "callback_input_api", "implementation_requirements",
        "side", "target_path", "host_state_work_order_scope",
    )
    task = {key: selected[key] for key in selected_fields if key in selected}
    previous_source = None
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            previous_source = parsed.get("implementation")
    except (TypeError, ValueError):
        pass
    task.update({
        "diagnostic": diagnostic,
        # Preserve byte-exact model output for trace/repro and existing repair ABI.
        "previous_response": response,
        "previous_implementation": previous_source if isinstance(previous_source, str) else response,
        "immutable_source_frame": dict(frame),
        "api_evidence": context,
        "correction_obligation": (
            "Change the actual invalid Java expression or statement. Repeating "
            "the same candidate does not resolve the diagnostic. Implement the "
            "same behavior using verified APIs, without TODOs or dropping effects. "
            "Return exactly one JSON string field named implementation."
        ),
    })
    return [
        {"role": "system", "content": (
            "Fix ONLY the Java declaration body rejected by the exact parser "
            "or state ABI diagnostic. The host fixes its signature and wrapper. "
            "If a Java grammar construct expects a type, supply a type; never "
            "reuse an expression in a type position. Preserve the feature behavior "
            "and all executable obligations. No imports, new classes or prose. "
            "Never replace missing readState_/writeState_ calls with an unrelated "
            "declared state key. If the required persistent state is missing "
            "from the canonical model, do not fabricate methods or defaults; "
            "use verified Minecraft world/event inputs only if they implement "
            "the original behavior."
        )},
        {"role": "user", "content": json.dumps(task, ensure_ascii=False, default=str)},
    ]


def preserve_failed_java(error: ValueError, candidate: object, *,
                         kind: str, name: str, owner: str, phase: str):
    from .first_pass_java_declarations import _record_source_failure

    source = candidate if isinstance(candidate, str) else getattr(error, "failed_source", None)
    if isinstance(source, str):
        _record_source_failure(source, kind=kind, name=name, owner=owner, phase=phase, exc=error)
    return source


def java_correction_message(error: ValueError, source: object, frame: Mapping,
                            *, feature: bool = False) -> dict[str, str]:
    correction = {
        ("repair_current_declaration" if feature else "repair_only_this_method"): True,
        ("error" if feature else "diagnostic"): str(error),
        ("previous_candidate" if feature else "previous_source"): source,
        "required_prefix": frame["prefix"], "required_suffix": frame["suffix"],
    }
    if feature:
        correction["required_signature"] = frame["prefix"]
        correction["instruction"] = (
            "Return JSON with one implementation string containing the corrected body; "
            "the host preserves the exact frame.")
    return {"role": "user", "content": json.dumps(correction, ensure_ascii=False)}


def report_recoverable_direct_failure(symbol: str, error: ValueError) -> bool:
    from .java_region_parser import JavaRegionParseError

    recoverable = isinstance(error, JavaRegionParseError) or str(error).startswith((
        "CONSTRUCTION_DIRECT_TARGET_SYMBOL_INVALID",
        "CONSTRUCTION_DIRECT_FEATURE_TARGET_SYMBOL_INVALID",
        "CONSTRUCTION_DIRECT_BODY_INCOMPLETE",
        "CONSTRUCTION_DIRECT_SIGNATURE_INVALID",
        "CONSTRUCTION_DIRECT_BODY_NOT_VALIDATED",
        "CONSTRUCTION_UNIT_",
        "CONSTRUCTION_SOURCE_SIGNATURE_INVALID",
        "CONSTRUCTION_SOURCE_IMPLEMENTATION_INVALID",
    ))
    if recoverable:
        print("CONSTRUCTION_AUTHOR_ROUTE_SWITCH: " + json.dumps({
            "member": symbol, "from": "direct_java", "to": "typed_ir",
            "reason": "direct_source_validation_failed",
            "diagnostic": str(error)[:1200],
        }, ensure_ascii=False), flush=True)
    return recoverable
