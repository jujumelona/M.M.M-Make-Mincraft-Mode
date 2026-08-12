from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any, Sequence


_FULL_PAGE_DECODE_LIMIT = 2


def _latest_saved_stream(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    latest = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    break
                if isinstance(value, dict) and isinstance(value.get("text"), str):
                    latest = value["text"]
    except OSError:
        return ""
    return latest


class _BoundedPageDecodeRouter:
    """Limit only expensive full production-page decodes, not child repairs."""

    def __init__(self, router: Any, error_type: type[Exception]) -> None:
        self._router = router
        self._error_type = error_type
        self._full_page_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def generate_text(self, *args: Any, **kwargs: Any) -> Any:
        messages = args[1] if len(args) > 1 else kwargs.get("messages")
        role = args[0] if args else kwargs.get("role")
        response_format = kwargs.get("response_format")
        system = ""
        if isinstance(messages, (list, tuple)) and messages:
            first = messages[0]
            if isinstance(first, dict):
                system = str(first.get("content", ""))
        full_page = (
            role == "planner"
            and response_format == "json"
            and "HOST JSON CONTRACT: Return production JSON" in system
        )
        if full_page:
            self._full_page_calls += 1
            if self._full_page_calls > _FULL_PAGE_DECODE_LIMIT:
                raise self._error_type(
                    "Production page failed after one page-local repair; "
                    "saved stream/item repair remains authoritative."
                )
        return self._router.generate_text(*args, **kwargs)


def install(complete_planner_module: Any) -> None:
    """Replay fsynced streams and bound expensive full-page regeneration.

    The first malformed full page may receive exactly one page-local retry. Further
    recovery must use the durable stream and item-level semantic repair instead of an
    unbounded sequence of fresh model decodes.
    """

    from . import production_stream_efficiency_contract as stream

    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_saved_stream_resume", False):
        return

    @wraps(current)
    def generate_with_saved_stream_resume(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        production = (
            isinstance(request, dict)
            and len(expected_contracts) == 1
            and expected_contracts[0]
            == frozenset(complete_planner_module._PRODUCTION_PAGE_CONTRACT)
        )
        if production:
            from . import planner_json_runtime_contract as runtime

            saved_text = _latest_saved_stream(stream._stream_event_path(stage, request))
            if saved_text:
                page = stream._salvage_production_stream(
                    complete_planner_module,
                    runtime,
                    router,
                    text=saved_text,
                    request=request,
                    stage=stage,
                )
                if page is not None:
                    return page

            router = _BoundedPageDecodeRouter(
                router,
                complete_planner_module.SpecValidationError,
            )

        return current(
            router,
            system_prompt=system_prompt,
            request=request,
            media_paths=media_paths,
            expected_contracts=expected_contracts,
            stage=stage,
        )

    generate_with_saved_stream_resume._mmm_saved_stream_resume = True  # type: ignore[attr-defined]
    generate_with_saved_stream_resume._mmm_bounded_full_page_decode = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_saved_stream_resume


__all__ = ["install", "_BoundedPageDecodeRouter"]
