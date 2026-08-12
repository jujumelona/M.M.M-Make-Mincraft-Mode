from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any, Sequence


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
                    # Ignore a torn final append; earlier fsynced events remain valid.
                    break
                if isinstance(value, dict) and isinstance(value.get("text"), str):
                    latest = value["text"]
    except OSError:
        return ""
    return latest


def install(complete_planner_module: Any) -> None:
    """Replay the last fsynced failed production stream before any new decode.

    Once a saved stream contains salvageable production semantics, recovery is the
    authoritative path. Backend/process errors must propagate so the same durable
    fragment remains the restart point; they must never trigger a fresh full-page GPU
    decode that discards already-generated work.
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
                # Do not catch failures here. Semantic child-repair errors are handled
                # inside the stream salvage loops. A router/backend/process failure is
                # intentionally surfaced while the fsynced raw stream and repair state
                # remain untouched for the next run.
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

        # A new full-page decode is allowed only when there is no saved stream or the
        # saved stream contains no host-verifiable production semantic object at all.
        return current(
            router,
            system_prompt=system_prompt,
            request=request,
            media_paths=media_paths,
            expected_contracts=expected_contracts,
            stage=stage,
        )

    generate_with_saved_stream_resume._mmm_saved_stream_resume = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_saved_stream_resume


__all__ = ["install"]
