from __future__ import annotations

"""Host-owned fixed-template generation for every structured model response.

Real generation adapters fill forced function arguments. The model never authors JSON
serialization syntax. The deterministic ``mock`` profile keeps its fixture transport only
so fast tests can remain model-free.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .model_output_atomicity_contract import assert_atomic_model_schema
from .structured_output import validate_structured_output

_JSON_FIXTURE_FORMAT = "json"
_ROLE_TOOL_STAGE = {
    "planner": "planning",
    "researcher": "research",
    "coder": "generation",
    "coder_safe": "quality",
    "visual_critic": "quality",
}
_DEFAULT_TOOL_NAME = "submit_fixed_template"


def _adapter_name(router: Any, role: str) -> str:
    try:
        config = router.registry.role(router.profile, role)
    except Exception:
        return ""
    return str(getattr(config, "adapter", "") or "")


def _semantic_prelude_required(
    router: Any,
    role: str,
    *,
    media_paths: Sequence[str | Path],
    tool_stage: str | None,
    enable_tools: bool,
) -> bool:
    if media_paths:
        return True
    if not enable_tools:
        return False
    enabled = getattr(router, "_tools_enabled", None)
    if not callable(enabled):
        return False
    stage = str(tool_stage or _ROLE_TOOL_STAGE.get(role, "") or "").strip().lower()
    return bool(
        enabled(
            enable_tools=True,
            stage=stage,
            adapter_name=_adapter_name(router, role),
        )
    )


def _tool_parameters(response_schema: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    if response_schema.get("type") == "object" or "properties" in response_schema:
        return dict(response_schema), False
    return (
        {
            "type": "object",
            "properties": {"value": dict(response_schema)},
            "required": ["value"],
            "additionalProperties": False,
        },
        True,
    )


def _template_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    semantic_output: str,
) -> tuple[dict[str, Any], ...]:
    copied = tuple(dict(message) for message in messages)
    if not semantic_output.strip():
        return copied
    return (
        *copied,
        {
            "role": "system",
            "content": (
                "The semantic/tool/media pass is complete. Use the result only as evidence "
                "when filling the supplied fixed template. Do not reproduce serialization "
                "syntax or protocol prose.\n\nCompleted semantic result:\n"
                + semantic_output
            ),
        },
    )


def generate_fixed_template_value(
    router: Any,
    role: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    response_schema: Mapping[str, Any],
    media_paths: Sequence[str | Path] = (),
    tool_stage: str | None = None,
    enable_tools: bool = True,
    tool_name: str = _DEFAULT_TOOL_NAME,
    description: str = "",
) -> Any:
    """Return host-validated structured data without model-authored JSON syntax."""

    if not isinstance(response_schema, Mapping):
        raise TypeError("fixed-template generation requires a response_schema mapping")
    assert_atomic_model_schema(response_schema, surface=f"fixed template for role {role!r}")

    # ``mock`` is a deterministic fixture engine, not a model. Preserve its existing
    # schema-aware fixture transport without providing this escape hatch to real adapters.
    if _adapter_name(router, role) == "mock":
        raw = router.generate_text(
            role,
            messages,
            media_paths=media_paths,
            response_format=_JSON_FIXTURE_FORMAT,
            response_schema=response_schema,
            tool_stage=tool_stage,
            enable_tools=enable_tools,
        )
        validated = validate_structured_output(
            raw,
            response_format=_JSON_FIXTURE_FORMAT,
            response_schema=response_schema,
        )
        return json.loads(validated)

    semantic_output = ""
    if _semantic_prelude_required(
        router,
        role,
        media_paths=media_paths,
        tool_stage=tool_stage,
        enable_tools=enable_tools,
    ):
        semantic_output = router.generate_text(
            role,
            messages,
            media_paths=media_paths,
            response_format="text",
            response_schema=None,
            tool_stage=tool_stage,
            enable_tools=enable_tools,
        )

    parameters, unwrap_value = _tool_parameters(response_schema)
    arguments = router.generate_tool_decision(
        role,
        _template_messages(messages, semantic_output=semantic_output),
        tool_name=str(tool_name or _DEFAULT_TOOL_NAME),
        parameters=parameters,
        description=(
            description.strip()
            or "Fill the host-supplied fixed response template exactly once. Populate only declared fields."
        ),
    )
    value: Any
    if unwrap_value:
        if "value" not in arguments:
            raise ValueError("fixed-template function call omitted wrapped value")
        value = arguments["value"]
    else:
        value = arguments

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    validated = validate_structured_output(
        encoded,
        response_format=_JSON_FIXTURE_FORMAT,
        response_schema=response_schema,
    )
    return json.loads(validated)


def generate_fixed_template_text(
    router: Any,
    role: str,
    messages: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> str:
    """Compatibility surface returning host-serialized JSON after fixed-template fill."""

    value = generate_fixed_template_value(router, role, messages, **kwargs)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = ["generate_fixed_template_text", "generate_fixed_template_value"]
