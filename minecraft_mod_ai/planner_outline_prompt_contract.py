from __future__ import annotations

from functools import wraps
from typing import Any, Sequence


_OUTLINE_FIELDS = frozenset({"production_batches", "complete", "next_cursor"})

_SCALABLE_OUTLINE_PROMPT = """You are the production-outline JSON API for a Minecraft mod planner.
Return only production-outline JSON pages; no Markdown, analysis, commentary, examples, or echoed contract text.

Every page is one complete JSON object with exactly these top-level keys:
production_batches, complete, next_cursor.
Every production batch must use exactly the batch fields supplied by the host contract.

Choose page size yourself from the actual plan complexity and the available model context/output budget.
There is NO fixed batch count and NO fixed page count.
Do not pad, duplicate, or artificially split small work, but do not truncate a large plan to fit one object either.

If the whole outline fits comfortably, return one JSON object with complete=true and next_cursor="".
If more work remains, close the current JSON object cleanly with complete=false and a short non-empty next_cursor.
You may then either emit the next complete JSON page immediately or stop and let the host request continuation.
If several JSON objects are emitted in one response, they are consecutive pages of ONE outline in order. The host preserves every valid page and uses the final emitted page to decide whether another request is needed.

On a repair request, the host may provide accepted_outline_prefix containing pages/batches already validated and saved. NEVER regenerate, rename, summarize, or replace those accepted batches. Continue only from the first rejected/missing part described by repair_error.
Never leave a JSON object truncated. Prefer another page over an oversized or incomplete object.
"""


def _outline_is_allowed(expected_contracts: Sequence[frozenset[str]]) -> bool:
    return _OUTLINE_FIELDS in tuple(expected_contracts)


def install(runtime_module: Any) -> None:
    """Give every production-outline path model-chosen, unbounded pagination."""

    from . import complete_planner as complete_planner_module

    page_current = complete_planner_module._generate_json_page_with_repair
    if getattr(page_current, "_mmm_scalable_outline_prompt", False):
        return

    @wraps(page_current)
    def generate_outline_with_scalable_prompt(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        if not _outline_is_allowed(expected_contracts):
            return page_current(
                router,
                system_prompt=system_prompt,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )

        return page_current(
            router,
            system_prompt=_SCALABLE_OUTLINE_PROMPT,
            request=request,
            media_paths=media_paths,
            expected_contracts=expected_contracts,
            stage=stage,
        )

    generate_outline_with_scalable_prompt._mmm_scalable_outline_prompt = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_outline_with_scalable_prompt


__all__ = ["install"]
