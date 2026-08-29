from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import validators
from jsonschema.exceptions import SchemaError


class StructuredOutputValidationError(RuntimeError):
    """Raised when one completed model response violates its JSON contract."""

    def __init__(
        self,
        *,
        output: str,
        errors: Sequence[str],
    ) -> None:
        self.output = output
        self.errors = tuple(errors)
        self.repair_attempts = 0
        details = "; ".join(self.errors)
        super().__init__(f"structured output is invalid: {details}")


def _json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f"[{json.dumps(str(part), ensure_ascii=False)}]"
    return path


def _validator_for(schema: Mapping[str, Any]):
    schema_dict = dict(schema)
    validator_cls = validators.validator_for(schema_dict)
    try:
        validator_cls.check_schema(schema_dict)
    except SchemaError as exc:
        raise ValueError(f"invalid response_schema: {exc.message}") from exc
    return validator_cls(schema_dict)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _emit_validation_failure(
    *,
    output: str,
    errors: Sequence[str],
    response_format: str,
    response_schema: Mapping[str, Any] | None,
) -> None:
    """Print the complete failed model response and validation contract immediately.

    Debugging a local agent must not collapse a concrete parse/schema failure into a later
    ``failure_count``. This event is emitted at the first authority that has all three facts:
    the exact model text, the exact validation errors, and the exact schema used for the turn.
    The raw output is intentionally preserved verbatim in the JSON event; its SHA makes it
    easy to correlate the same response with downstream research/checkpoint diagnostics.
    """

    payload = {
        "event": "structured_output_validation_failure",
        "response_format": response_format,
        "output_sha256": _sha256_text(output),
        "output_chars": len(output),
        "output": output,
        "errors": list(errors),
        "response_schema": dict(response_schema) if response_schema is not None else None,
    }
    print(
        "MODEL STRUCTURED OUTPUT FAILURE: "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def validate_structured_output(
    output: str,
    *,
    response_format: str,
    response_schema: Mapping[str, Any] | None,
) -> str:
    """Validate one final response without extraction, coercion, or regeneration.

    ``response_format='json'`` requires exactly one valid JSON document. When a
    response schema is supplied it is applied to that decoded JSON value, including
    valid scalar, array, object, and null roots. The original model text is returned
    unchanged after validation so this boundary never rewrites semantic output.
    """

    if response_schema is None and response_format != "json":
        return output
    if response_schema is not None and not isinstance(response_schema, Mapping):
        raise ValueError("response_schema must be a mapping")

    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        errors = (
            f"$: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        )
        _emit_validation_failure(
            output=output,
            errors=errors,
            response_format=response_format,
            response_schema=response_schema,
        )
        raise StructuredOutputValidationError(
            output=output,
            errors=errors,
        ) from exc

    if response_schema is None:
        return output

    validator = _validator_for(response_schema)
    validation_errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    if validation_errors:
        errors = tuple(
            f"{_json_path(tuple(error.absolute_path))}: {error.message}"
            for error in validation_errors
        )
        _emit_validation_failure(
            output=output,
            errors=errors,
            response_format=response_format,
            response_schema=response_schema,
        )
        raise StructuredOutputValidationError(
            output=output,
            errors=errors,
        )
    return output


__all__ = [
    "StructuredOutputValidationError",
    "validate_structured_output",
]
