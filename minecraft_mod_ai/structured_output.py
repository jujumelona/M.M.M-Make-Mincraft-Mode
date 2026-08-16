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


def _repair_truncated_json(text: str) -> str:
    """Structurally repair JSON truncated by max_tokens.

    Walks the text tracking string/escape state and a delimiter stack.
    If the text ends inside a string literal, the string is closed.
    Any remaining open ``{`` / ``[`` delimiters are then closed in
    correct nesting order so the result is syntactically valid JSON.

    This is a HOST-SIDE structural repair — no model round-trip needed.
    The upper pagination layer will see a partial-but-valid page and
    generate the remaining content via continuation.
    """
    in_string = False
    escaped = False
    stack: list[str] = []  # expected closing delimiters in nesting order

    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    suffix = ""
    if in_string:
        suffix += '"'
    while stack:
        suffix += stack.pop()
    return text + suffix


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
        r'(?<= ")(.*?)(?=")',
        lambda m: m.group(1).replace("\n", "\\n").replace("\r", "").replace("\t", "\\t"),
        cleaned,
        flags=re.DOTALL,
    )
    try:
        return json.loads(repaired, strict=False), repaired, None
    except json.JSONDecodeError:
        pass

    # Layer 5: truncation repair — close unterminated strings & delimiters
    truncation_repaired = _repair_truncated_json(cleaned)
    try:
        return json.loads(truncation_repaired, strict=False), truncation_repaired, None
    except json.JSONDecodeError:
        pass

    # Layer 6: both control-char repair AND truncation repair combined
    combined = _repair_truncated_json(repaired)
    try:
        return json.loads(combined, strict=False), combined, None
    except json.JSONDecodeError:
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


def _coerce_value_to_schema(value: Any, schema: dict[str, Any], *, aggressive: bool = False) -> Any:
    """Coerce a JSON value to conform to a schema fragment."""
    if not isinstance(schema, dict):
        return value

    target_type = schema.get("type")
    
    # Handle array type
    if target_type == "array":
        item_schema = schema.get("items", {})
        if isinstance(value, (list, tuple)):
            coerced_items = [_coerce_value_to_schema(item, item_schema, aggressive=aggressive) for item in value]
            min_items = schema.get("minItems", 0)
            if len(coerced_items) < min_items and aggressive:
                while len(coerced_items) < min_items:
                    coerced_items.append(_coerce_value_to_schema("", item_schema, aggressive=aggressive))
            return coerced_items
        elif isinstance(value, str):
            val_str = value.strip()
            if val_str:
                return [_coerce_value_to_schema(val_str, item_schema, aggressive=aggressive)]
            return []
        elif isinstance(value, (int, float, bool)):
            return [_coerce_value_to_schema(str(value), item_schema, aggressive=aggressive)]
        elif isinstance(value, dict):
            # If an object was provided where an array is expected, wrap or extract values
            return [_coerce_value_to_schema(value, item_schema, aggressive=aggressive)]
        return []

    # Handle object type
    if target_type == "object":
        if not isinstance(value, dict):
            if isinstance(value, str) and value.strip():
                value = {"summary": value.strip()}
            else:
                value = {}
        result = dict(value)
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in result:
                result[prop_name] = _coerce_value_to_schema(result[prop_name], prop_schema, aggressive=aggressive)
            elif aggressive and prop_name in schema.get("required", []):
                result[prop_name] = _coerce_value_to_schema(None, prop_schema, aggressive=aggressive)
        
        # Additional properties check
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for k in list(result):
                if k not in properties:
                    result[k] = _coerce_value_to_schema(result[k], additional, aggressive=aggressive)
        elif additional is False:
            for k in list(result):
                if k not in properties:
                    del result[k]
        return result

    # Handle string type
    if target_type == "string":
        if isinstance(value, str):
            min_len = schema.get("minLength", 0)
            if not value.strip() and min_len > 0:
                return "default" if aggressive else value
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value if v is not None)
        if value is None:
            return "default" if aggressive else ""
        return str(value)

    # Handle integer / number type
    if target_type in ("integer", "number"):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value) if target_type == "integer" else float(value)
        if isinstance(value, str):
            try:
                num = float(value.strip())
                return int(num) if target_type == "integer" else num
            except (ValueError, TypeError):
                pass
        return 0 if target_type == "integer" else 0.0

    # Handle boolean type
    if target_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    return value


def _coerce_json_to_schema(json_text: str, schema: Mapping[str, Any], *, aggressive: bool = False) -> str | None:
    """Safely parse JSON text, coerce structure to match schema, and re-encode."""
    try:
        data, _, exc = _parse_json_value(json_text)
    except Exception:
        return None

    if exc is not None or not isinstance(data, (dict, list)):
        return None

    try:
        coerced_data = _coerce_value_to_schema(data, dict(schema), aggressive=aggressive)
        return json.dumps(coerced_data, ensure_ascii=False)
    except Exception:
        return None


def _recover_invalid_json_document(exc: ModelBackendError) -> str | None:
    """Recover model JSON rejected at the host syntax boundary, never transport JSON.

    The early-stop transport may close a structurally complete root object before a
    Python syntax check. In that case its direct ModelBackendError cause is the
    JSONDecodeError raised by json.loads(content), whose ``doc`` is the exact model
    output. SSE/protocol failures are wrapped in another RuntimeError, so they do not
    match this narrow recovery path and remain transport failures.

    If the raw document is truncated (e.g. max_tokens reached), it is auto-repaired
    so that the validation layer receives syntactically valid JSON and the upper
    pagination layer can treat a partial page as a valid continuation point.
    """

    cause = exc.cause
    if not isinstance(cause, json.JSONDecodeError):
        # Also handle RuntimeError wrapping (e.g. from _parse_root_json_object)
        if isinstance(cause, RuntimeError) and cause.__cause__:
            cause = cause.__cause__
        if not isinstance(cause, json.JSONDecodeError):
            return None
    output = str(cause.doc).strip()
    cleaned = _clean_json_text(output)
    if not cleaned.startswith("{") and "{" in cleaned:
        cleaned = cleaned[cleaned.find("{"):]
    if not cleaned.startswith("{"):
        return None
    # Auto-repair truncated JSON so validation gets syntactically valid input
    repaired = _repair_truncated_json(cleaned)
    try:
        json.loads(repaired, strict=False)
        return repaired  # Return the repaired version, not the broken original
    except json.JSONDecodeError:
        return output  # Fallback: return raw and let _parse_json_value handle it


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

    # Attempt deterministic host schema coercion on candidate JSON
    coerced = _coerce_json_to_schema(current, effective_schema)
    if coerced is not None:
        coerced_errors = _validation_errors(coerced, validator)
        if not coerced_errors:
            return coerced

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
        coerced = _coerce_json_to_schema(current, effective_schema)
        if coerced is not None:
            coerced_errors = _validation_errors(coerced, validator)
            if not coerced_errors:
                return coerced

    # Final aggressive deterministic coercion fallback
    final_coerced = _coerce_json_to_schema(current, effective_schema, aggressive=True)
    if final_coerced is not None:
        final_errors = _validation_errors(final_coerced, validator)
        if not final_errors:
            return final_coerced

    raise StructuredOutputValidationError(
        output=current,
        errors=errors,
        repair_attempts=max_repair_attempts,
    )


__all__ = [
    "StructuredOutputValidationError",
    "generate_with_host_schema_repair",
]