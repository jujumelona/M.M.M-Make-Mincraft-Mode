from __future__ import annotations

"""Single host-owned output-budget policy for text generation backends.

A configured ``max_new_tokens`` can be a request-packing reservation without becoming
an artificial decode ceiling. Profiles opt into that behavior with
``dynamic_output_budget``. Compact reviewed tool calls stay finitely bounded, while
source mutation/generation actions may use the live context that remains after the
current request.
"""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from .model_context_budget import effective_context_tokens, tool_action_token_budget

_CONTEXT_GUARD_TOKENS = 2048
_BYTES_PER_TOKEN_ESTIMATE = 2
_DEFAULT_DYNAMIC_OUTPUT_TOKENS = 16384
_EXPANSIVE_TOOL_EFFECTS = frozenset(
    {
        "project_changed",
        "source_generated",
        "assets_generated",
        "generated",
        "repaired",
        "packaged",
    }
)


def _positive_override(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def dynamic_output_budget_enabled(config: Any) -> bool:
    extra = getattr(config, "extra", {}) or {}
    if not isinstance(extra, Mapping):
        return False
    value = extra.get("dynamic_output_budget", False)
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _configured_output_ceiling(config: Any) -> int | None:
    override = _positive_override("MMM_GENERATION_MAX_TOKENS")
    if override is not None:
        return override

    adapter = str(getattr(config, "adapter", "") or "").strip().casefold()
    if adapter in {"llama_cpp", "vllm"}:
        legacy = _positive_override("MMM_LLAMA_TEXT_MAX_TOKENS")
        if legacy is not None:
            return legacy

    if dynamic_output_budget_enabled(config):
        return None
    try:
        configured = int(getattr(config, "max_new_tokens", 0) or 0)
    except (TypeError, ValueError):
        configured = 0
    return configured if configured > 0 else None


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, Mapping):
        to_schema = getattr(tool, "to_schema", None)
        if callable(to_schema):
            tool = to_schema()
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", "") or "").strip()
    return str(tool.get("name", "") or "").strip()


def tools_require_expansive_output(tools: Sequence[Any]) -> bool:
    """Return true unless every visible tool is reviewed as a compact action."""

    if not tools:
        return False
    from .tool_transition_registry import reviewed_transition

    for tool in tools:
        name = _tool_name(tool)
        if not name:
            return True
        transition = reviewed_transition(name)
        if transition is None:
            return True
        if transition.effects & _EXPANSIVE_TOOL_EFFECTS:
            return True
    return False


def generation_output_token_budget(
    config: Any,
    *,
    input_tokens: int = 0,
    tools: Sequence[Any] = (),
) -> int:
    """Return one finite decode budget without imposing the compact-tool cap globally."""

    ceiling = _configured_output_ceiling(config)
    context = effective_context_tokens(config)
    if ceiling is not None:
        budget = ceiling
    elif context > 0:
        budget = max(1, context - max(0, int(input_tokens)) - _CONTEXT_GUARD_TOKENS)
    else:
        budget = _DEFAULT_DYNAMIC_OUTPUT_TOKENS

    if tools and not tools_require_expansive_output(tools):
        budget = min(budget, tool_action_token_budget(config))
    return max(1, int(budget))


def payload_input_token_estimate(payload: Mapping[str, Any]) -> int:
    """Conservatively estimate serialized input tokens for transport-level clamping."""

    value = dict(payload)
    value.pop("max_tokens", None)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except Exception:
        return 0
    return max(0, (len(encoded) + _BYTES_PER_TOKEN_ESTIMATE - 1) // _BYTES_PER_TOKEN_ESTIMATE)


def apply_payload_generation_budget(
    payload: Mapping[str, Any],
    *,
    config: Any,
) -> dict[str, Any]:
    """Apply the common finite output budget to one OpenAI-compatible payload."""

    bounded = dict(payload)
    raw_tools = bounded.get("tools")
    tools: Sequence[Any] = (
        tuple(raw_tools)
        if isinstance(raw_tools, Sequence)
        and not isinstance(raw_tools, (str, bytes, bytearray))
        else ()
    )
    input_tokens = payload_input_token_estimate(bounded)
    budget = generation_output_token_budget(
        config,
        input_tokens=input_tokens,
        tools=tools,
    )

    context = effective_context_tokens(config)
    if context > 0:
        remaining = max(1, context - input_tokens - _CONTEXT_GUARD_TOKENS)
        budget = min(budget, remaining)

    try:
        requested = int(bounded.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        requested = 0
    if requested > 0 and not dynamic_output_budget_enabled(config):
        budget = min(budget, requested)

    bounded["max_tokens"] = max(1, int(budget))
    return bounded


__all__ = [
    "apply_payload_generation_budget",
    "dynamic_output_budget_enabled",
    "generation_output_token_budget",
    "payload_input_token_estimate",
    "tools_require_expansive_output",
]
