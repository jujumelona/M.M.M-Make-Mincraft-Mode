from __future__ import annotations

import json
from functools import wraps
from typing import Any, Sequence


def _unwrap_transport_json_fence(text: str) -> str:
    """Remove one response-wide ```json fence and nothing else."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    newline = stripped.find("\n")
    if newline < 0:
        return stripped
    opener = stripped[:newline].strip().casefold()
    if opener != "```json" or not stripped.endswith("```"):
        return stripped

    return stripped[newline + 1 : -3].strip()


def _unwrap_qwen_think_channel(text: str) -> str:
    """Remove one leading Qwen reasoning-channel wrapper, if present.

    The pinned Colab OpenAI-compatible server can leave Qwen's chat-template
    ``<think>...</think>`` delimiter in ``delta.content`` even when host policy has
    disabled reasoning.  A template can also pre-open the channel and expose only a
    leading ``</think>`` before visible content.  These are transport/channel markers,
    not planner output.  Only one *leading* channel wrapper is ignored; arbitrary
    prose before/after the JSON remains invalid and an unterminated think block is
    deliberately left untouched so strict JSON parsing rejects it.
    """

    stripped = text.strip()
    if stripped.startswith("<think>"):
        close = stripped.find("</think>", len("<think>"))
        if close < 0:
            return stripped
        return stripped[close + len("</think>") :].strip()
    if stripped.startswith("</think>"):
        return stripped[len("</think>") :].strip()
    return stripped


def _normalize_structured_transport(text: str) -> str:
    """Normalize only known structured-output transport wrappers.

    This is intentionally not a JSON recovery routine: it never searches for an
    embedded object, auto-closes JSON, deletes prose, or accepts multiple values.
    After the known Qwen channel marker and optional response-wide JSON fence are
    removed, the complete remainder still has to pass ``json.loads`` exactly once.
    """

    candidate = text.strip()
    # A BOM is a serialization marker rather than model prose.  Accept at most one at
    # the absolute start, which also keeps ``json.loads`` behavior deterministic.
    if candidate.startswith("\ufeff"):
        candidate = candidate[1:].lstrip()
    candidate = _unwrap_qwen_think_channel(candidate)
    candidate = _unwrap_transport_json_fence(candidate)
    return candidate


def install(runtime_module: Any) -> None:
    """Require one complete JSON object on structured planner pages.

    The planner already has a bounded page-local retry. Accepting embedded prose,
    ``strict=False`` JSON, or auto-closed truncated objects bypasses that repair gate
    and can synthesize pagination state the model never actually completed. Structured
    decode output therefore has one contract: one complete strict JSON object whose
    top-level field set matches one declared host contract. Known Colab/Qwen transport
    wrappers are normalized before that strict boundary is enforced.
    """

    current = runtime_module._extract_with_safe_empty_defaults
    if getattr(current, "_mmm_strict_structured_json", False):
        return

    @wraps(current)
    def extract_strict(
        module: Any,
        text: str,
        *,
        expected_contracts: Sequence[frozenset[str]],
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise module.SpecValidationError("Structured planner returned empty JSON.")
        candidate = _normalize_structured_transport(text)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise module.SpecValidationError(
                f"Structured planner did not return one complete strict JSON object: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise module.SpecValidationError(
                "Structured planner top-level JSON value must be an object."
            )
        fields = frozenset(str(key) for key in value)
        if fields not in tuple(expected_contracts):
            expected = [sorted(contract) for contract in expected_contracts]
            raise module.SpecValidationError(
                "Structured planner top-level fields do not match the host contract: "
                f"received={sorted(fields)}, expected_one_of={expected}"
            )
        return dict(value)

    extract_strict._mmm_strict_structured_json = True  # type: ignore[attr-defined]
    extract_strict.__wrapped__ = current  # type: ignore[attr-defined]
    runtime_module._extract_with_safe_empty_defaults = extract_strict


__all__ = ["install"]
