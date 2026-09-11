from __future__ import annotations

"""Deterministic host-side template renderer for Java, JSON, and resource molds."""

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .template_contract_validation import PLACEHOLDER, validate_template_contract
from .version_template_context import resolved_template_values


class TemplateRenderError(ValueError):
    pass


_PLACEHOLDER_PATTERN = PLACEHOLDER


def _substitute_string(template_str: str, values: Mapping[str, Any]) -> str:
    def replacer(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise TemplateRenderError(
                f"RENDER_MISSING_VALUE: Required placeholder '{{{{{key}}}}}' is missing from values"
            )
        return str(values[key])

    return _PLACEHOLDER_PATTERN.sub(replacer, template_str)


def _substitute_json_data(data: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(data, str):
        match = _PLACEHOLDER_PATTERN.fullmatch(data)
        if match and match.group(1) in values:
            return deepcopy(values[match.group(1)])
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
    validate_template_contract(template)
    render_values = resolved_template_values(values)
    if "inputs" in template:
        contracts = template["inputs"]
        schema = {
            "type": "object",
            "properties": {
                name: {key: value for key, value in spec.items() if key != "required"}
                for name, spec in contracts.items()
            },
            "required": [name for name, spec in contracts.items() if spec.get("required") is True],
            "additionalProperties": False,
        }
        # The execution context may carry unrelated deterministic facts. Validate only the
        # inputs this template declares; required version inputs can be satisfied by HOST
        # injection above, while undeclared execution metadata never becomes model input.
        contract_values = {
            name: render_values[name]
            for name in contracts
            if name in render_values
        }
        try:
            Draft202012Validator(schema).validate(contract_values)
        except Exception as exc:
            raise TemplateRenderError(f"RENDER_INPUT_CONTRACT: {exc}") from exc
    render_spec = template.get("render")
    if render_spec is None:
        raise TemplateRenderError(
            f"RENDER_NO_SPEC: Template {template.get('id', '<unknown>')} declares no 'render' section"
        )
    if isinstance(render_spec, str):
        return _substitute_string(render_spec, render_values)
    if not isinstance(render_spec, Mapping):
        raise TemplateRenderError(
            "RENDER_INVALID_SPEC: 'render' section must be a string or mapping"
        )

    language = str(render_spec.get("language", "text")).lower()
    body = render_spec.get("body")
    if body is None:
        raise TemplateRenderError("RENDER_NO_BODY: 'render' section must declare 'body'")

    if language == "json":
        if isinstance(body, (Mapping, list)):
            substituted = _substitute_json_data(body, render_values)
            return json.dumps(
                substituted, indent=2, sort_keys=True, ensure_ascii=False
            ) + "\n"
        if isinstance(body, str):
            rendered_str = _substitute_string(body, render_values)
            try:
                parsed = json.loads(rendered_str)
            except Exception as exc:
                raise TemplateRenderError(
                    f"RENDER_INVALID_JSON: Rendered JSON is invalid: {exc}"
                ) from exc
            return json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        raise TemplateRenderError(
            "RENDER_INVALID_JSON_BODY: JSON body must be mapping, list, or string"
        )

    if isinstance(body, str):
        return _substitute_string(body, render_values)
    raise TemplateRenderError(
        f"RENDER_UNSUPPORTED_LANGUAGE: Language {language!r} is not supported"
    )
