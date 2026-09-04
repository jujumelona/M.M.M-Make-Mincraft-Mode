from __future__ import annotations

"""Single host-owned output-budget policy for text generation backends.

A configured ``max_new_tokens`` can be a request-packing reservation without becoming
an artificial decode ceiling. Profiles opt into that behavior with
``dynamic_output_budget``. Compact reviewed tool calls and host-selected structured
planning decisions stay finitely bounded, while source mutation/generation actions may
use the live context that remains after the current request.

Transport token estimation is intentionally conservative, but it is only an estimate.
Dynamic forced structural actions retain a generous 4096-token page so an estimated
context squeeze cannot silently starve them. Explicit static ceilings remain
host-authoritative, but the host refuses genuinely non-viable structural decodes before
inference instead of sending the 0/1-token requests that caused the continuation loop.
"""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from .model_context_budget import effective_context_tokens, tool_action_token_budget

_CONTEXT_GUARD_TOKENS = 2048
# Qwen's live prompt in production measured materially above two serialized bytes/token.
# Three remains conservative while avoiding the 2x overestimate that reduced a normal
# apply_source_edit turn to 523 tokens and its retry to 151 tokens.
_BYTES_PER_TOKEN_ESTIMATE = 3
_DEFAULT_DYNAMIC_OUTPUT_TOKENS = 16384
_MIN_STRUCTURAL_TOOL_OUTPUT_TOKENS = 4096
# This is a liveness/safety floor, not a quality target. Deliberate static profiles may
# choose 512/1024-token scalar edit pages; accidental context collapse to 0/1 (or another
# tiny fragment) must fail before inference. Dynamic profiles still receive the larger
# 4096-token structural floor above.
_MIN_VIABLE_STRUCTURAL_TOOL_OUTPUT_TOKENS = 128
_STRUCTURAL_COMPACT_TOOLS = frozenset({"apply_source_edit"})
# These are not executable agent actions. They are host-selected, schema-constrained
# semantic decisions whose arguments are consumed and validated by the host. Treating an
# unregistered decision function as an unknown side-effecting tool previously let it
# inherit nearly the whole 32k runtime context (~30k output tokens). Reuse the existing
# bounded function-call page budget instead of inventing a planner-specific token number.
_HOST_STRUCTURED_DECISION_TOOLS = frozenset(
    {"compile_semantic_requirements"}
)
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


class GenerationOutputBudgetError(RuntimeError):
    """Raised before inference when one complete host-required action cannot fit."""



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


def _structural_tool_call_is_compact(tools: Sequence[Any]) -> bool:
    """Classify serialized call shape, independently from the tool's side effects."""

    if not tools:
        return False
    names = {_tool_name(tool) for tool in tools}
    return bool(names and names <= _STRUCTURAL_COMPACT_TOOLS)


def _structural_tool_floor(config: Any, tools: Sequence[Any]) -> int:
    if _structural_tool_call_is_compact(tools):
        return max(
            1,
            min(
                _MIN_STRUCTURAL_TOOL_OUTPUT_TOKENS,
                tool_action_token_budget(config),
            ),
        )
    return 1


def _assert_structural_budget_viable(
    config: Any,
    tools: Sequence[Any],
    budget: int,
    *,
    source: str,
) -> None:
    """Fail before a structurally useless fragment reaches model inference.

    The 4096-token structural floor is a *dynamic packing target*. It must not override a
    deliberate static profile ceiling such as 512/1024. The separate viability floor is
    intentionally much smaller and exists only to reject collapsed fragments that cannot
    reasonably encode one scalar tool action.
    """

    if not _structural_tool_call_is_compact(tools):
        return
    viable = min(
        _MIN_VIABLE_STRUCTURAL_TOOL_OUTPUT_TOKENS,
        max(1, tool_action_token_budget(config)),
    )
    if int(budget) >= viable:
        return
    names = sorted({_tool_name(tool) for tool in tools if _tool_name(tool)})
    raise GenerationOutputBudgetError(
        "STRUCTURAL_OUTPUT_BUDGET_UNVIABLE: host-required tool action "
        f"{names or ['<unknown>']} needs at least {viable} output tokens, but {source} "
        f"permits only {max(0, int(budget))}; refusing a partial inference request."
    )


def tools_require_expansive_output(tools: Sequence[Any]) -> bool:
    """Classify tool effects, not how many tokens its function arguments need.

    ``apply_source_edit`` remains expansive here because it mutates project source and
    preflight/recovery policy depends on that effect classification. Its scalar protocol
    is nevertheless a compact *call shape* (one type shell, import, or member); output
    budgeting handles that separately by guaranteeing a minimum page, not by imposing
    the compact-tool maximum.

    Host-selected semantic/retrieval decision functions are intentionally different:
    they have no side effect, execute no external action, and are fully schema validated.
    They therefore use the already-reviewed finite tool page budget even though they are
    not entries in the executable transition registry.
    """

    if not tools:
        return False
    from .tool_transition_registry import reviewed_transition

    for tool in tools:
        name = _tool_name(tool)
        if not name:
            return True
        if name in _HOST_STRUCTURED_DECISION_TOOLS:
            continue
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
    """Return one finite decode budget without starving a forced structural action."""

    ceiling = _configured_output_ceiling(config)
    context = effective_context_tokens(config)
    floor = _structural_tool_floor(config, tools)

    if ceiling is not None:
        # Explicit static ceilings are host-authoritative. We reject only a truly
        # non-viable fragment; we do not silently promote 512/1024 to the dynamic floor.
        _assert_structural_budget_viable(
            config,
            tools,
            ceiling,
            source="configured output ceiling",
        )
        budget = ceiling
    elif context > 0:
        estimated_remaining = (
            context - max(0, int(input_tokens)) - _CONTEXT_GUARD_TOKENS
        )
        budget = max(floor, estimated_remaining, 1)
    else:
        budget = max(floor, _DEFAULT_DYNAMIC_OUTPUT_TOKENS)

    # Only genuinely non-expansive effects receive the compact action ceiling. Source
    # mutation is allowed to use the remaining context; its scalar protocol is protected
    # by the structural minimum above rather than by a small hard maximum.
    if tools and not tools_require_expansive_output(tools):
        budget = min(budget, tool_action_token_budget(config))

    _assert_structural_budget_viable(
        config,
        tools,
        int(budget),
        source="computed output budget",
    )
    return max(1, int(budget))


def payload_input_token_estimate(payload: Mapping[str, Any]) -> int:
    """Estimate serialized input tokens for transport packing, not hard context truth."""

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
    except (TypeError, ValueError, OverflowError, RecursionError):
        return 0
    return max(
        0,
        (len(encoded) + _BYTES_PER_TOKEN_ESTIMATE - 1)
        // _BYTES_PER_TOKEN_ESTIMATE,
    )


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
        # Serialized-size estimation is deliberately not allowed to squeeze one forced
        # source edit below the amount needed to finish its function arguments. If the
        # real server context is tighter, llama_finish_reason_contract reports typed
        # CONTEXT_PRESSURE and the canonical tool loop compacts observations.
        floor = _structural_tool_floor(config, tools)
        estimated_remaining = context - input_tokens - _CONTEXT_GUARD_TOKENS
        remaining = max(floor, estimated_remaining, 1)
        budget = min(budget, remaining)

    try:
        requested = int(bounded.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        requested = 0
    if requested > 0 and not dynamic_output_budget_enabled(config):
        _assert_structural_budget_viable(
            config,
            tools,
            requested,
            source="request max_tokens",
        )
        budget = min(budget, requested)

    _assert_structural_budget_viable(
        config,
        tools,
        int(budget),
        source="final payload budget",
    )
    bounded["max_tokens"] = max(1, int(budget))
    return bounded


__all__ = [
    "GenerationOutputBudgetError",
    "apply_payload_generation_budget",
    "dynamic_output_budget_enabled",
    "generation_output_token_budget",
    "payload_input_token_estimate",
    "tools_require_expansive_output",
]
