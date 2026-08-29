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


def _parser_owned_research_schema(schema: Mapping[str, Any] | None) -> bool:
    """Identify the intentionally loose envelope whose semantics are host-parser owned."""

    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        return False
    if schema.get("required"):
        return False
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return False
    note = properties.get("research_note")
    return (
        isinstance(note, Mapping)
        and note.get("type") == "object"
        and not note.get("required")
        and note.get("additionalProperties") is True
    )


def _extract_embedded_object(output: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _emit_validation_failure(
    *,
    output: str,
    errors: Sequence[str],
    response_format: str,
    response_schema: Mapping[str, Any] | None,
) -> None:
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


def _emit_parser_owned_recovery(output: str, value: Mapping[str, Any]) -> None:
    print(
        "MODEL STRUCTURED OUTPUT RECOVERED: "
        + json.dumps(
            {
                "event": "parser_owned_embedded_json_recovery",
                "output_sha256": _sha256_text(output),
                "output_chars": len(output),
                "output": output,
                "recovered_value": dict(value),
                "authority": "research_host_parser",
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def _schema_errors(
    value: Any,
    response_schema: Mapping[str, Any],
) -> tuple[str, ...]:
    validator = _validator_for(response_schema)
    validation_errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    return tuple(
        f"{_json_path(tuple(error.absolute_path))}: {error.message}"
        for error in validation_errors
    )


def validate_structured_output(
    output: str,
    *,
    response_format: str,
    response_schema: Mapping[str, Any] | None,
) -> str:
    """Validate transport shape without rejecting output the designated host parser owns.

    Normal JSON contracts remain strict. The deliberately loose research-note envelope is
    different: its downstream parser already locates the first JSON object and canonicalizes
    Qwen variants. For that one contract, surrounding prose must not trigger an expensive
    full model regeneration. The embedded object is schema-checked here, the original text
    is returned unchanged, and the host parser remains the semantic authority.
    """

    if response_schema is None and response_format != "json":
        return output
    if response_schema is not None and not isinstance(response_schema, Mapping):
        raise ValueError("response_schema must be a mapping")

    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        if _parser_owned_research_schema(response_schema):
            embedded = _extract_embedded_object(output)
            if embedded is not None:
                errors = _schema_errors(embedded, response_schema)
                if not errors:
                    _emit_parser_owned_recovery(output, embedded)
                    return output
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

    errors = _schema_errors(value, response_schema)
    if errors:
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
