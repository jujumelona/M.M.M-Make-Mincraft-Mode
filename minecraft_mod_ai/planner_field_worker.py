"""One content field per model call; validation, retries and receipts stay on the host."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .planner_stage_trace import PlannerStageTrace
from .spec import SpecValidationError


def generate_field(
    router: Any,
    *,
    messages: Sequence[Mapping[str, str]],
    parse: Callable[[str], Any],
    trace: PlannerStageTrace,
    field: str,
    requirement_ref: str = "",
    media_paths: Sequence[Any] = (),
) -> Any:
    history = [dict(message) for message in messages]
    for attempt in (1, 2):
        raw = ""
        try:
            raw = router.generate_text(
                "planner",
                history,
                media_paths=media_paths,
                response_format="text",
                response_schema=None,
                tool_stage="game_design",
                enable_tools=False,
            )
            value = parse(str(raw or ""))
        except (SpecValidationError, ValueError, TypeError, KeyError) as exc:
            trace.record_attempt(
                raw_output=str(raw),
                validation_error=f"{type(exc).__name__}: {exc}",
                context={
                    "field": field,
                    "requirement_ref": requirement_ref,
                    "attempt": attempt,
                },
            )
            if attempt == 2:
                raise SpecValidationError(
                    f"Design field {field!r} for {requirement_ref or 'request'} remains unresolved: {exc}"
                ) from exc
            history.extend(
                [
                    {"role": "assistant", "content": str(raw)},
                    {
                        "role": "user",
                        "content": (
                            f"Repair only field {field}. Validation failed: {exc}. "
                            "Preserve its valid content. Return only this field's content, no analysis or unrelated fields."
                        ),
                    },
                ]
            )
            continue
        except Exception as exc:
            trace.record_attempt(
                raw_output=str(raw),
                validation_error=f"{type(exc).__name__}: {exc}",
                context={
                    "field": field,
                    "requirement_ref": requirement_ref,
                    "attempt": attempt,
                },
            )
            raise
        trace.record_attempt(
            raw_output=str(raw),
            validation_error=None,
            accepted={field: value},
            context={
                "field": field,
                "requirement_ref": requirement_ref,
                "attempt": attempt,
            },
        )
        return value
    raise AssertionError("unreachable field retry state")
