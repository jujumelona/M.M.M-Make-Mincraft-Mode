from __future__ import annotations

from functools import wraps
from typing import Any, Sequence


_OUTLINE_FIELDS = frozenset({"production_batches", "complete", "next_cursor"})

_SCALABLE_OUTLINE_PROMPT = """You are the production-outline JSON API for a Minecraft mod planner.
Return only production-outline JSON pages; no Markdown, analysis, commentary, examples, or echoed contract text.

Every page is one complete JSON object with exactly these top-level keys:
production_batches, complete, next_cursor.
Every production batch must use exactly the batch fields supplied by the host contract.

Choose the page size yourself from the actual plan complexity and the available model context/output budget.
There is NO fixed batch count and NO fixed page count.
Do not pad, duplicate, or artificially split small work, but do not truncate a large plan to fit one object either.

If the whole outline fits comfortably, return one JSON object with complete=true and next_cursor="".
If more work remains, close the current JSON object cleanly with complete=false and a short non-empty next_cursor.
You may then either:
1. emit the next complete JSON page immediately after it, or
2. stop after that page and let the host request the continuation using next_cursor.
If you emit multiple JSON objects in one response, they are consecutive pages of ONE outline, in order; every non-final page must have complete=false and a non-empty next_cursor. The final emitted page carries the true complete/next_cursor state for the remaining outline.

Never leave a JSON object truncated. Prefer another page over an oversized or incomplete object.
"""


def _is_outline_contract(expected_contracts: Sequence[frozenset[str]]) -> bool:
    return len(expected_contracts) == 1 and expected_contracts[0] == _OUTLINE_FIELDS


def install(runtime_module: Any) -> None:
    """Give production outlines a scalable, model-chosen pagination prompt.

    This installer deliberately does not change ``max_tokens``, batch count, page count,
    or retry count.  The selected model keeps its configured output budget and chooses
    how much outline work belongs in each page.  Large plans scale through explicit
    ``complete``/``next_cursor`` pagination instead of host-imposed size limits.
    """

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
        if not _is_outline_contract(expected_contracts):
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
