from __future__ import annotations

"""Deterministic host-side template renderer for Java, JSON, and resource molds.

Takes a template specification and concrete values, performing exact substitution
without calling any language model.
"""

import json
import re
from collections.abc import Mapping
from typing import Any


class TemplateRenderError(ValueError):
    pass


_PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")


def _substitute_string(template_str: str, values: Mapping[str, Any]) -> str:
    def replacer(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise TemplateRenderError(
                f"RENDER_MISSING_VALUE: Required placeholder '{{{{{key}}}}}' is missing from values"
            )
        val = values[key]
        return str(val)

    return _PLACEHOLDER_PATTERN.sub(replacer, template_str)


def _substitute_json_data(data: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(data, str):
        return _substitute_string(data, values)
    if isinstance(data, Mapping):
        return {
            _substitute_string(str(k), values): _substitute_json_data(v, values)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_substitute_json_data(item, values) for item in data]
    return data


def render_template(template: Mapping[str, Any], values: Mapping[str, Any]) -> str:
    """Render a template specification using deterministic substitution."""
    render_spec = template.get("render")
    if render_spec is None:
        raise TemplateRenderError(
            f"RENDER_NO_SPEC: Template {template.get('id', '<unknown>')} declares no 'render' section"
        )

    # If raw string
    if isinstance(render_spec, str):
        return _substitute_string(render_spec, values)

    if not isinstance(render_spec, Mapping):
        raise TemplateRenderError("RENDER_INVALID_SPEC: 'render' section must be a string or mapping")

    language = str(render_spec.get("language", "text")).lower()
    body = render_spec.get("body")
    if body is None:
        raise TemplateRenderError("RENDER_NO_BODY: 'render' section must declare 'body'")

    if language == "json":
        if isinstance(body, (Mapping, list)):
            substituted = _substitute_json_data(body, values)
            return json.dumps(substituted, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        elif isinstance(body, str):
            rendered_str = _substitute_string(body, values)
            try:
                parsed = json.loads(rendered_str)
                return json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            except Exception as exc:
                raise TemplateRenderError(f"RENDER_INVALID_JSON: Rendered JSON is invalid: {exc}") from exc
        else:
            raise TemplateRenderError("RENDER_INVALID_JSON_BODY: JSON body must be mapping, list, or string")

    # Java or plain text
    if isinstance(body, str):
        return _substitute_string(body, values)

    raise TemplateRenderError(f"RENDER_UNSUPPORTED_LANGUAGE: Language {language!r} is not supported")
