from __future__ import annotations

"""Deterministic host-side template renderer for Java, JSON, and resource molds."""

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .template_contract_validation import PLACEHOLDER, validate_template_contract


class TemplateRenderError(ValueError):
    pass


_PLACEHOLDER_PATTERN = PLACEHOLDER


def _resolved_render_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Inject immutable HOST target facts without allowing caller overrides.

    Templates receive the complete resolved target automatically whenever execution carries
    ``resolved_version_context``.  This keeps version-coupled constants out of model inputs
    and prevents individual templates from rebuilding compatibility rules independently.
    """
    merged = dict(values)
    raw = merged.get("resolved_version_context")
    if raw is None:
        return merged

    from .resolved_version_context import ResolvedVersionContext, VersionContextError
    from .target_contract import mappings_applicable

    resolved = raw if isinstance(raw, ResolvedVersionContext) else ResolvedVersionContext.from_dict(raw)
    snapshot = resolved.to_dict()
    target = dict(snapshot["target"])

    # Derived target facts are deterministic too.  Expose them from the same authority so
    # templates never need to infer naming/pack semantics from a Minecraft version string.
    mapping_is_applicable = mappings_applicable(target["minecraft_version"])
    canonical = {
        **target,
        "version_context_id": resolved.context_id,
        "mappings_applicable": mapping_is_applicable,
        "naming_regime": "mapped_obfuscated" if mapping_is_applicable else "native_unobfuscated",
        "pack_versions": {
            "data": target["data_pack_version"],
            "resource": target["resource_pack_version"],
            "resource_major": target["resource_pack_format"],
        },
    }
    if mapping_is_applicable:
        canonical["mappings"] = {
            "kind": target["mappings_kind"],
            "version": target["mappings_version"],
        }

    for key, expected in canonical.items():
        if key in merged and merged[key] != expected:
            raise VersionContextError(
                "HOST_FACT_OVERRIDE",
                field=key,
                expected=expected,
                actual=merged[key],
                context_id=resolved.context_id,
            )
        if key not in merged:
            merged[key] = deepcopy(expected)
    return merged


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
    render_values = _resolved_render_values(values)
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
        # The execution context may carry unrelated deterministic facts.  Validate only the
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
