"""Preserve failed Java bytes and isolate direct-source recovery decisions."""
from __future__ import annotations

import json
from collections.abc import Mapping


def assemble_java_implementation(response: str, frame: Mapping) -> str:
    """Decode implementation data once; only the host writes declaration syntax."""
    from .model_adapters.base import ModelConfigurationError

    try:
        value = json.loads(response)
        if (not isinstance(value, dict) or set(value) != {"implementation"}
                or not isinstance(value["implementation"], str)):
            raise ValueError("expected exactly one implementation string")
    except (ValueError, TypeError) as cause:
        error = ModelConfigurationError(
            "CONSTRUCTION_SOURCE_IMPLEMENTATION_INVALID: " + str(cause))
        error.failed_response = response
        raise error from cause
    body = value["implementation"]
    source = (frame["empty_value"] if frame.get("kind") == "field"
              and body == "" and "empty_value" in frame
              else frame["prefix"] + body + frame["suffix"])
    try:
        return validate_generated_java_source(
            source, frame, frame["prefix"], frame["suffix"])
    except ValueError as error:
        error.failed_response = response
        raise


def validate_generated_java_source(result, frame: Mapping, prefix: str, suffix: str) -> str:
    from .model_adapters.base import ModelConfigurationError
    from .first_pass_java_declarations import validate_source_frame

    if (not isinstance(result, str) or not (
            result == frame.get("empty_value")
            or (result.startswith(prefix) and result.endswith(suffix)))):
        error = ModelConfigurationError("CONSTRUCTION_SOURCE_SIGNATURE_INVALID")
        error.failed_source = result
        raise error
    try:
        validate_source_frame(result, frame)
    except ValueError as error:
        error.failed_source = result
        raise
    return result


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
    ))
    if recoverable:
        print("CONSTRUCTION_AUTHOR_ROUTE_SWITCH: " + json.dumps({
            "member": symbol, "from": "direct_java", "to": "typed_ir",
            "reason": "direct_source_validation_failed",
            "diagnostic": str(error)[:1200],
        }, ensure_ascii=False), flush=True)
    return recoverable
