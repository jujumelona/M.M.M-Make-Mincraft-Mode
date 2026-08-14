from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from jsonschema import validators
from jsonschema.exceptions import SchemaError

from .model_adapters import GenerationRequest


_MAX_REPAIR_ATTEMPTS = 2


class StructuredOutputValidationError(RuntimeError):
    """Raised after bounded correction passes cannot satisfy the response schema."""

    def __init__(
        self,
        *,
        output: str,
        errors: Sequence[str],
        repair_attempts: int,
    ) -> None:
        self.output = output
        self.errors = tuple(errors)
        self.repair_attempts = int(repair_attempts)
        details = "; ".join(self.errors)
        super().__init__(
            "structured output remained invalid after "
            f"{self.repair_attempts} repair attempt(s): {details}"
        )


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


def _validation_errors(text: str, validator: Any) -> tuple[str, ...]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return (
            f"$: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        )

    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    return tuple(
        f"{_json_path(tuple(error.absolute_path))}: {error.message}"
        for error in errors
    )


def _repair_messages(
    *,
    output: str,
    errors: Sequence[str],
    schema: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    error_text = "\n".join(f"- {error}" for error in errors)
    schema_text = json.dumps(
        dict(schema),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        {
            "role": "system",
            "content": (
                "You are a strict JSON repair engine. Repair the previous output; "
                "do not solve or regenerate the original task. Preserve every value "
                "and structure that already satisfies the schema. Change only what "
                "is required by the listed validation errors. Return only corrected "
                "JSON, with no markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Validation errors:\n{error_text}\n\n"
                f"Required JSON schema:\n{schema_text}\n\n"
                f"Previous invalid output:\n{output}"
            ),
        },
    )


def generate_with_host_schema_repair(
    request: GenerationRequest,
    generate: Callable[[GenerationRequest], str],
    *,
    max_repair_attempts: int = _MAX_REPAIR_ATTEMPTS,
) -> str:
    """Generate once, then correct only an existing invalid JSON result.

    Transport failures are deliberately not caught. Detailed JSON Schema enforcement
    belongs to this host boundary; the native server only needs to guarantee JSON
    syntax. Each correction pass receives the latest invalid output and its exact
    validation failures, and never replays the original task conversation.
    """

    schema = request.response_schema
    if request.response_format != "json" or not isinstance(schema, Mapping) or not schema:
        return generate(request)
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be non-negative")

    validator = _validator_for(schema)
    current = generate(request)
    errors = _validation_errors(current, validator)
    if not errors:
        return current

    for attempt in range(1, max_repair_attempts + 1):
        repair_request = replace(
            request,
            messages=_repair_messages(output=current, errors=errors, schema=schema),
            media_paths=(),
            response_format="json",
            response_schema=dict(schema),
            tools=(),
            tool_choice=None,
            parallel_tool_calls=False,
        )
        current = generate(repair_request)
        errors = _validation_errors(current, validator)
        if not errors:
            return current

    raise StructuredOutputValidationError(
        output=current,
        errors=errors,
        repair_attempts=max_repair_attempts,
    )


__all__ = [
    "StructuredOutputValidationError",
    "generate_with_host_schema_repair",
]
