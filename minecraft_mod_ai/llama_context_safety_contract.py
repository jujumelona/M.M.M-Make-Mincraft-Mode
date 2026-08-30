from __future__ import annotations

"""Hard context-capacity guard for local llama agent turns.

The normal context owner performs semantic compaction and archival. This final guard
exists for the cases where a minimum byte floor or an incompressible system/user core
would otherwise let an oversized request escape that owner. It never truncates the
original task or an applied source-mutation receipt silently: if the mandatory protocol
core cannot fit, the request fails before inference instead of entering repeated backend
context-pressure retries.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_MARKER = "_mmm_hard_context_capacity_v1"
_FIT_MARKER = "_mmm_hard_context_fit_v1"
_EMERGENCY_MARKER = "_mmm_protocol_safe_emergency_fit_v1"
_REPLACE_MARKER = "_mmm_protocol_safe_live_replace_v1"


class ContextPackingError(RuntimeError):
    """The mandatory live conversation cannot fit the active runtime slot."""


def _hard_llama_message_budget(
    context_module: Any,
    config: Any,
    tools: Sequence[Any],
) -> int | None:
    adapter = str(getattr(config, "adapter", "") or "").strip().casefold()
    if adapter != "llama_cpp":
        return None
    context = max(0, int(context_module.effective_context_tokens(config)))
    if context <= 0:
        return None
    try:
        max_input_tokens = max(0, int(getattr(config, "max_input_tokens", 0) or 0))
        max_new_tokens = max(0, int(getattr(config, "max_new_tokens", 0) or 0))
    except (TypeError, ValueError):
        max_input_tokens = 0
        max_new_tokens = 0

    reserved_output = (
        int(context_module.tool_action_token_budget(config)) if tools else max_new_tokens
    )
    available_tokens = max(0, context - max(0, reserved_output))
    if max_input_tokens > 0:
        available_tokens = min(available_tokens, max_input_tokens)
    available_tokens = max(
        0,
        available_tokens - int(context_module._CONTEXT_TOKEN_GUARD),
    )
    message_bytes = available_tokens * int(context_module._BYTES_PER_TOKEN_BUDGET)
    tool_bytes = context_module._canonical_size(tuple(tools)) if tools else 0
    return message_bytes - tool_bytes


def _leading_authority_indices(messages: Sequence[Mapping[str, Any]]) -> set[int]:
    """Keep every leading system instruction and the first authored user task."""

    selected: set[int] = set()
    index = 0
    while index < len(messages) and str(messages[index].get("role", "")) == "system":
        selected.add(index)
        index += 1
    if index < len(messages) and str(messages[index].get("role", "")) == "user":
        selected.add(index)
    return selected


def _latest_mutation_indices(messages: Sequence[Mapping[str, Any]]) -> set[int]:
    from .source_mutation_contract import mutation_observation_applied

    latest_tool: int | None = None
    for index, message in enumerate(messages):
        if mutation_observation_applied(message):
            latest_tool = index
    if latest_tool is None:
        return set()

    start = latest_tool
    for index in range(latest_tool - 1, -1, -1):
        message = messages[index]
        if str(message.get("role", "")) != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)):
            start = index
            break

    end = latest_tool + 1
    while end < len(messages) and str(messages[end].get("role", "")) == "tool":
        end += 1
    return set(range(start, end))


def _latest_protocol_tail_indices(
    messages: Sequence[Mapping[str, Any]],
    *,
    width: int,
) -> set[int]:
    if not messages:
        return set()
    lower = max(0, len(messages) - max(1, int(width)))
    start = len(messages) - 1
    for index in range(lower, len(messages)):
        role = str(messages[index].get("role", ""))
        if role in {"user", "assistant"}:
            start = index
            break
    return set(range(start, len(messages)))


def _protocol_safe_minimal_fit(
    context_module: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> tuple[Mapping[str, Any], ...]:
    """Shrink history without orphaning the task, mutation proof, or latest exchange."""

    original = tuple(dict(message) for message in messages)
    budget = max(1, int(budget))
    for width in (6, 4, 2, 1):
        selected = _leading_authority_indices(original)
        selected.update(_latest_mutation_indices(original))
        selected.update(_latest_protocol_tail_indices(original, width=width))
        candidate = tuple(original[index] for index in sorted(selected))
        candidate = context_module._compact_implementation_seed(candidate)
        candidate = context_module._compact_tool_messages(
            candidate,
            budget=budget,
            preview_bytes=2 * 1024,
        )
        if context_module._canonical_size(candidate) <= budget:
            return candidate

    mandatory = _leading_authority_indices(original)
    mandatory.update(_latest_mutation_indices(original))
    mandatory.update(_latest_protocol_tail_indices(original, width=1))
    candidate = tuple(original[index] for index in sorted(mandatory))
    candidate = context_module._compact_implementation_seed(candidate)
    candidate = context_module._compact_tool_messages(
        candidate,
        budget=budget,
        preview_bytes=2 * 1024,
    )
    size = context_module._canonical_size(candidate)
    if size <= budget:
        return candidate
    raise ContextPackingError(
        "mandatory agent context cannot fit the active llama runtime slot; "
        f"mandatory_bytes={size} budget_bytes={budget}. Refusing silent task/protocol truncation."
    )


def _is_unsafe_three_message_fallback(
    messages: Sequence[Mapping[str, Any]],
    replacement: Sequence[Mapping[str, Any]],
) -> bool:
    """Detect the legacy overflow fallback that drops middle authority/task messages."""

    if len(messages) <= 3 or len(replacement) != 3:
        return False
    return (
        replacement[0] == messages[0]
        and replacement[1] == messages[-2]
        and replacement[2] == messages[-1]
    )


def _install_tool_loop_guards(context_module: Any) -> None:
    # Import eagerly so the local names captured by progress_aware_tool_loop cannot
    # retain pre-guard context functions when runtime bootstrap finishes.
    from . import progress_aware_tool_loop as tool_loop

    tool_loop.request_message_budget = context_module.request_message_budget
    tool_loop.emergency_fit_messages = context_module.emergency_fit_messages
    tool_loop.fit_messages_to_context = context_module.fit_messages_to_context

    current_replace = tool_loop._replace_live_messages
    if getattr(current_replace, _REPLACE_MARKER, False):
        return

    @wraps(current_replace)
    def protocol_safe_live_replace(
        messages: list[dict[str, Any]],
        fitted: tuple[Mapping[str, Any], ...],
    ) -> bool:
        replacement = tuple(fitted)
        if _is_unsafe_three_message_fallback(messages, replacement):
            # Repeated backend context pressure used to discard every middle message,
            # including the original task after host-injected system instructions.
            # Keep no more bytes than that legacy triple, but preserve mandatory context.
            legacy_budget = max(1, context_module._canonical_size(replacement))
            replacement = _protocol_safe_minimal_fit(
                context_module,
                tuple(messages),
                budget=legacy_budget,
            )
        return current_replace(messages, replacement)

    setattr(protocol_safe_live_replace, _REPLACE_MARKER, True)
    protocol_safe_live_replace.__wrapped__ = current_replace  # type: ignore[attr-defined]
    tool_loop._replace_live_messages = protocol_safe_live_replace


def install(context_module: Any) -> None:
    """Install hard-capacity guards after the canonical semantic compaction owner."""

    current_request = context_module.request_message_budget
    if not getattr(current_request, _MARKER, False):

        @wraps(current_request)
        def safe_request_message_budget(config: Any, tools: Sequence[Any] = ()) -> int:
            policy_budget = max(1, int(current_request(config, tools)))
            hard_budget = _hard_llama_message_budget(context_module, config, tools)
            if hard_budget is None:
                return policy_budget
            if hard_budget <= 0:
                raise ContextPackingError(
                    "visible tools plus reserved output consume the active llama context; "
                    f"remaining_message_bytes={hard_budget}."
                )
            return max(1, min(policy_budget, int(hard_budget)))

        setattr(safe_request_message_budget, _MARKER, True)
        safe_request_message_budget.__wrapped__ = current_request  # type: ignore[attr-defined]
        context_module.request_message_budget = safe_request_message_budget

    current_emergency = context_module.emergency_fit_messages
    if not getattr(current_emergency, _EMERGENCY_MARKER, False):

        @wraps(current_emergency)
        def safe_emergency_fit_messages(
            messages: Sequence[Mapping[str, Any]],
            *,
            budget_bytes: int = 40 * 1024,
        ) -> tuple[Mapping[str, Any], ...]:
            exact_budget = max(
                1,
                min(int(context_module._MAX_CONTEXT_BYTES), int(budget_bytes)),
            )
            original = tuple(dict(message) for message in messages)
            original_size = context_module._canonical_size(original)
            fitted = current_emergency(original, budget_bytes=exact_budget)
            fitted_size = context_module._canonical_size(fitted)
            if fitted_size <= exact_budget and fitted_size < original_size:
                return tuple(fitted)

            # This path is entered after the backend has already proved context pressure.
            # If the byte estimator believed the unchanged payload fit, force a smaller
            # protocol-safe window rather than retrying the identical request.
            target = exact_budget
            if original_size <= exact_budget:
                target = max(1, min(exact_budget, int(original_size * 0.75)))
            return _protocol_safe_minimal_fit(
                context_module,
                fitted if fitted_size < original_size else original,
                budget=target,
            )

        setattr(safe_emergency_fit_messages, _EMERGENCY_MARKER, True)
        safe_emergency_fit_messages.__wrapped__ = current_emergency  # type: ignore[attr-defined]
        context_module.emergency_fit_messages = safe_emergency_fit_messages

    current_fit = context_module.fit_messages_to_context
    if not getattr(current_fit, _FIT_MARKER, False):

        @wraps(current_fit)
        def safe_fit_messages_to_context(
            messages: Sequence[Mapping[str, Any]],
            *,
            config: Any,
            tools: Sequence[Any] = (),
        ) -> tuple[Mapping[str, Any], ...]:
            budget = int(context_module.request_message_budget(config, tools))
            fitted = tuple(current_fit(messages, config=config, tools=tools))
            if context_module._canonical_size(fitted) <= budget:
                return fitted
            return _protocol_safe_minimal_fit(
                context_module,
                fitted,
                budget=budget,
            )

        setattr(safe_fit_messages_to_context, _FIT_MARKER, True)
        safe_fit_messages_to_context.__wrapped__ = current_fit  # type: ignore[attr-defined]
        context_module.fit_messages_to_context = safe_fit_messages_to_context

    _install_tool_loop_guards(context_module)


__all__ = [
    "ContextPackingError",
    "_hard_llama_message_budget",
    "_is_unsafe_three_message_fallback",
    "_protocol_safe_minimal_fit",
    "install",
]
