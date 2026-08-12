from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import Any, Sequence


_OUTLINE_FIELDS = frozenset({"production_batches", "complete", "next_cursor"})
_OUTLINE_TOKEN_CAP = 2048
_OUTLINE_MODE: ContextVar[bool] = ContextVar("mmm_outline_json_mode", default=False)

_COMPACT_OUTLINE_PROMPT = """You are a JSON API for a Minecraft mod production planner.
Return exactly ONE JSON object and nothing else.
The first non-whitespace character MUST be { and the last MUST be }.
Do not output Markdown fences, explanations, analysis, alternatives, examples, or a second JSON object.
Do not echo the request or the host contract.

The object has exactly these top-level keys: production_batches, complete, next_cursor.
production_batches must contain at most TWO new batches in this response.
Each batch must use the exact batch fields supplied by the host contract. Batch IDs must be new and unique relative to the host-provided catalogs.
Keep scope and deliverables concise but lossless; do not repeat the same requirement in multiple phrasings.
If more outline work remains after these batches, set complete=false and next_cursor to a short non-empty opaque string.
If the outline is finished, set complete=true and next_cursor="".
Never emit more than one top-level JSON object.
"""


def _is_outline_contract(expected_contracts: Sequence[frozenset[str]]) -> bool:
    return len(expected_contracts) == 1 and expected_contracts[0] == _OUTLINE_FIELDS


def install(runtime_module: Any) -> None:
    """Make small production-outline pages decode as small JSON pages.

    The planner role may allow 8k output tokens for implementation pages. An outline
    page has only three top-level fields and should never inherit that ceiling: doing
    so lets a model ramble through repeated candidate objects until max_tokens. This
    contract changes the prompt and request-local token ceiling only for the exact
    production-outline contract. Semantic validation remains owned by the existing
    strict planner JSON layer.
    """

    from . import complete_planner as complete_planner_module
    from . import llama_server_hardware_policy as hardware_policy_module

    payload_current = hardware_policy_module._server_payload
    if not getattr(payload_current, "_mmm_outline_token_cap", False):

        @wraps(payload_current)
        def payload_with_outline_cap(adapter: Any, request: Any) -> dict[str, Any]:
            payload = payload_current(adapter, request)
            if _OUTLINE_MODE.get():
                configured = int(payload.get("max_tokens", _OUTLINE_TOKEN_CAP))
                payload["max_tokens"] = min(configured, _OUTLINE_TOKEN_CAP)
            return payload

        payload_with_outline_cap._mmm_outline_token_cap = True  # type: ignore[attr-defined]
        hardware_policy_module._server_payload = payload_with_outline_cap

    budget_current = runtime_module._attempt_budget
    if not getattr(budget_current, "_mmm_outline_attempt_budget", False):

        @wraps(budget_current)
        def budget_with_outline_limit(production_page: bool) -> int:
            if _OUTLINE_MODE.get():
                # One clean decode plus one bounded correction. If this fails, the
                # prompt/output contract is broken; repeating the same call is waste.
                return 2
            return budget_current(production_page)

        budget_with_outline_limit._mmm_outline_attempt_budget = True  # type: ignore[attr-defined]
        runtime_module._attempt_budget = budget_with_outline_limit

    page_current = complete_planner_module._generate_json_page_with_repair
    if getattr(page_current, "_mmm_compact_outline_prompt", False):
        return

    @wraps(page_current)
    def generate_outline_with_compact_prompt(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        if not _is_outline_contract(expected_contracts):
            return page_current(
                router,
                system_prompt=system_prompt,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )

        token = _OUTLINE_MODE.set(True)
        try:
            return page_current(
                router,
                system_prompt=_COMPACT_OUTLINE_PROMPT,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )
        finally:
            _OUTLINE_MODE.reset(token)

    generate_outline_with_compact_prompt._mmm_compact_outline_prompt = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_outline_with_compact_prompt


__all__ = ["install"]
