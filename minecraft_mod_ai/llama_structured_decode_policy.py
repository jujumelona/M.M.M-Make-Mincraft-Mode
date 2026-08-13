from __future__ import annotations

import os
from functools import wraps
from typing import Any, Mapping


_MARKER = "_mmm_host_validated_json_fastpath_v5"


def _bounded_section_output_tokens(adapter: Any) -> int:
    configured = max(1, int(getattr(adapter.config, "max_new_tokens", 1) or 1))
    raw = os.environ.get("MMM_LLAMA_BOUNDED_SECTION_MAX_TOKENS", "").strip()
    try:
        requested = int(raw) if raw else 2048
    except ValueError:
        requested = 2048
    if requested <= 0:
        requested = 2048
    return min(configured, requested)


def _is_qwen35(adapter: Any) -> bool:
    config = getattr(adapter, "config", None)
    model_id = str(getattr(config, "model_id", "")).casefold()
    extra = getattr(config, "extra", {})
    filename = (
        str(extra.get("gguf_filename", "")).casefold()
        if isinstance(extra, Mapping)
        else ""
    )
    return "qwen3.5-9b" in model_id and ("mtp" in model_id or "mtp" in filename)


def _force_server_schema() -> bool:
    raw = os.environ.get("MMM_QWEN35_FORCE_SERVER_JSON_SCHEMA", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def bind_structured_decode_policy(hardware_module: Any) -> None:
    """Keep validated planner serialization off the constrained decode hot path.

    Schema-less JSON is parsed and contract-validated by MMM on the host. The Qwen
    game-design stage has the same explicit host validator/repair loop, so its large
    schema also defaults to ordinary decoding instead of a llama.cpp grammar. Other
    explicit schemas remain constrained until their host validation paths are proven.
    """

    current = hardware_module._server_payload
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = current(adapter, request)
        if result.get("tools"):
            return result
        if getattr(request, "response_format", None) != "json":
            return result

        schema = getattr(request, "response_schema", None)
        if schema is None:
            # Host-validated planner pages do not need llama.cpp's JSON grammar.
            result.pop("response_format", None)
            result["reasoning_effort"] = "none"
            result["chat_template_kwargs"] = {"enable_thinking": False}
            result.pop("thinking_budget_tokens", None)
            return result

        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        if (
            isinstance(properties, Mapping)
            and "game_design" in properties
            and _is_qwen35(adapter)
            and not _force_server_schema()
        ):
            # GameDesignPlanner validates every candidate after generation and
            # repairs invalid output until success or a proven no-progress cycle.
            # Avoid converting this large schema to a sampler grammar on the T4
            # speculative-decode path; the host validator remains authoritative.
            result.pop("response_format", None)
            result["reasoning_effort"] = "none"
            result["chat_template_kwargs"] = {"enable_thinking": False}
            result.pop("thinking_budget_tokens", None)
            return result

        if isinstance(properties, Mapping) and "section" in properties:
            # This is a transport budget for one explicitly bounded serialization
            # call, not a project/plan-size or input-context limit. Large/schema-less
            # paginated JSON retains the model profile's full output budget.
            current_max = max(1, int(result.get("max_tokens", 1) or 1))
            result["max_tokens"] = min(
                current_max,
                _bounded_section_output_tokens(adapter),
            )
            result["thinking_budget_tokens"] = 0
            result["reasoning_effort"] = "none"
        return result

    setattr(payload, _MARKER, True)
    payload._mmm_bounded_section_thinking_budget_v2 = True  # type: ignore[attr-defined]
    payload._mmm_bounded_section_thinking_budget_v1 = True  # type: ignore[attr-defined]
    payload._mmm_qwen35_game_design_host_validation = True  # type: ignore[attr-defined]
    hardware_module._server_payload = payload


__all__ = ["bind_structured_decode_policy"]
