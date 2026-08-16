from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from jsonschema import validators
from jsonschema.exceptions import SchemaError

from .model_adapters import GenerationRequest, ModelBackendError


_MAX_REPAIR_ATTEMPTS = 2
_GENERIC_JSON_OBJECT_SCHEMA: Mapping[str, Any] = {"type": "object"}


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


import re


def _clean_json_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    elif "<think>" in cleaned:
        if "{" in cleaned:
            cleaned = cleaned[cleaned.find("{"):]
        else:
            cleaned = ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _parse_json_value(text: str) -> tuple[Any, str, json.JSONDecodeError | None]:
    """Parse JSON with multi-layer sanitization for model-generated output."""
    raw = text.strip()
    try:
        return json.loads(raw), raw, None
    except json.JSONDecodeError as direct_err:
        last_err = direct_err

    try:
        return json.loads(raw, strict=False), raw, None
    except json.JSONDecodeError:
        pass

    cleaned = _clean_json_text(raw)
    try:
        return json.loads(cleaned, strict=False), cleaned, None
    except json.JSONDecodeError:
        pass

    # Sanitize unescaped control chars / newlines inside string literals
    repaired = re.sub(
        r'(?<=: ")(.*?)(?=")',
        lambda m: m.group(1).replace("\n", "\\n").replace("\r", "").replace("\t", "\\t"),
        cleaned,
        flags=re.DOTALL,
    )
    try:
        return json.loads(repaired, strict=False), repaired, None
    except json.JSONDecodeError as final_err:
        return None, raw, last_err


def _validation_errors(text: str, validator: Any) -> tuple[str, ...]:
    value, _, exc = _parse_json_value(text)
    if exc is not None or value is None:
        if exc is None:
            return ("$: invalid JSON: unable to parse document",)
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


def _recover_invalid_json_document(exc: ModelBackendError) -> str | None:
    """Recover model JSON rejected at the host syntax boundary, never transport JSON.

    The early-stop transport may close a structurally complete root object before a
    Python syntax check. In that case its direct ModelBackendError cause is the
    JSONDecodeError raised by json.loads(content), whose ``doc`` is the exact model
    output. SSE/protocol failures are wrapped in another RuntimeError, so they do not
    match this narrow recovery path and remain transport failures.
    """

    cause = exc.cause
    if not isinstance(cause, json.JSONDecodeError):
        return None
    output = str(cause.doc).strip()
    cleaned = _clean_json_text(output)
    if not cleaned.startswith("{") and "{" in cleaned:
        cleaned = cleaned[cleaned.find("{"):]
    if not cleaned.startswith("{"):
        return None
    return output


def _generate_json_candidate(
    request: GenerationRequest,
    generate: Callable[[GenerationRequest], str],
) -> str:
    try:
        return generate(request)
    except ModelBackendError as exc:
        recovered = _recover_invalid_json_document(exc)
        if recovered is None:
            raise
        return recovered


def generate_with_host_schema_repair(
    request: GenerationRequest,
    generate: Callable[[GenerationRequest], str],
    *,
    max_repair_attempts: int = _MAX_REPAIR_ATTEMPTS,
) -> str:
    """Generate JSON, then correct only an existing invalid JSON result.

    Detailed JSON Schema enforcement belongs to this host boundary. For historical
    schema-less JSON calls, a successful backend return is intentionally left alone;
    callers may use that mode as a transport hint. The only schema-less value that is
    intercepted is a structurally complete model JSON document that the strict native
    server has already rejected with a direct JSONDecodeError. That exact document is
    then repaired against the generic object grammar.

    Genuine transport failures are never converted into repair work. Each correction
    pass receives only the latest invalid output and exact validation failures, not the
    original task conversation. Supplied schemas are validated before model execution
    so a host configuration error can never consume a model request.
    """

    if request.response_format != "json":
        return generate(request)
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be non-negative")

    schema = request.response_schema
    if schema is None:
        try:
            return generate(request)
        except ModelBackendError as exc:
            current = _recover_invalid_json_document(exc)
            if current is None:
                raise
        effective_schema = dict(_GENERIC_JSON_OBJECT_SCHEMA)
        validator = _validator_for(effective_schema)
    elif isinstance(schema, Mapping):
        effective_schema = dict(schema) if schema else dict(_GENERIC_JSON_OBJECT_SCHEMA)
        validator = _validator_for(effective_schema)
        current = _generate_json_candidate(request, generate)
    else:
        raise ValueError("response_schema must be a mapping when response_format='json'")

    errors = _validation_errors(current, validator)
    if not errors:
        return current

    for _attempt in range(1, max_repair_attempts + 1):
        repair_request = replace(
            request,
            messages=_repair_messages(
                output=current,
                errors=errors,
                schema=effective_schema,
            ),
            media_paths=(),
            response_format="json",
            response_schema=effective_schema,
            tools=(),
            tool_choice=None,
            parallel_tool_calls=False,
        )
        current = _generate_json_candidate(repair_request, generate)
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
