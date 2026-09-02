from __future__ import annotations

"""Reconcile model-output and official-template boundaries after runtime composition.

The central contracts remain fail-closed. This late adapter only removes bounded model
scaffolding before the existing design validator runs and ensures Fabric's official
bootstrap cannot overwrite the approval-bound platform receipt with a legacy narrow lock.
"""

import json
import re
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALLED = False

_THINK_BLOCK_RE = re.compile(
    r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>",
    re.IGNORECASE | re.DOTALL,
)
_IDENTITY_WRAPPERS = {
    "title": (
        re.compile(
            r"^\s*(?:here(?:'s| is)|below is)\s+(?:the\s+)?title\s*[:\-–—]\s*",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*(?:the\s+)?title\s*(?:is|:)\s*", re.IGNORECASE),
    ),
    "pitch": (
        re.compile(
            r"^\s*(?:here(?:'s| is)|below is)\s+(?:the\s+)?pitch\s*[:\-–—]\s*",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*(?:the\s+)?pitch\s*(?:is|:)\s*", re.IGNORECASE),
    ),
}


def _sanitize_design_body(raw: Any, field: str) -> str:
    """Remove only balanced hidden-reasoning blocks and narrow identity wrappers.

    Unbalanced ``<think>`` markers are deliberately left untouched so the existing
    ``assert_design_field_clean`` guard still rejects/falls back rather than guessing where
    model reasoning ended. Balanced blocks are safe to remove because their boundaries are
    explicit; all authored bullets before/after the block are preserved verbatim.
    """

    body = str(raw or "")
    body = _THINK_BLOCK_RE.sub("", body)
    if field in _IDENTITY_WRAPPERS:
        stripped = body.strip()
        for pattern in _IDENTITY_WRAPPERS[field]:
            candidate = pattern.sub("", stripped, count=1).strip()
            if candidate != stripped:
                stripped = candidate
                break
        body = stripped
    return body


def _write_approval_bound_bootstrap_lock(
    root: Path,
    adapter: Any,
    receipt: dict[str, Any],
) -> None:
    """Use the canonical immutable writer, then attach non-authoritative bootstrap evidence."""

    from . import platform_generation_contract

    # immutable_platform_execution_contract.install() replaces this callable with the
    # canonical v4 writer before execution. Delegating here prevents the official Fabric
    # provider from reintroducing its historical narrow v2 lock.
    platform_generation_contract._write_platform_lock(root, adapter)
    target = Path(root) / ".minecraft_ai" / "platform-lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Canonical platform lock writer did not produce an object.")
    payload["bootstrap"] = dict(receipt)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import agentic_research_game_design as agentic
    from . import fabric_official_template_provider as fabric_provider

    original_parse_field_output = agentic._parse_field_output
    if not getattr(original_parse_field_output, "_mmm_meta_sanitized_boundary", False):

        @wraps(original_parse_field_output)
        def parse_field_output(raw: Any, field: str) -> Any:
            return original_parse_field_output(_sanitize_design_body(raw, field), field)

        parse_field_output._mmm_meta_sanitized_boundary = True
        parse_field_output.__wrapped__ = original_parse_field_output
        agentic._parse_field_output = parse_field_output

    original_platform_lock_writer = fabric_provider._write_platform_lock
    if not getattr(original_platform_lock_writer, "_mmm_approval_bound_bootstrap_lock", False):

        @wraps(original_platform_lock_writer)
        def write_platform_lock(root: Path, adapter: Any, receipt: dict[str, Any]) -> None:
            _write_approval_bound_bootstrap_lock(root, adapter, receipt)

        write_platform_lock._mmm_approval_bound_bootstrap_lock = True
        write_platform_lock.__wrapped__ = original_platform_lock_writer
        fabric_provider._write_platform_lock = write_platform_lock

    _INSTALLED = True


__all__ = ["install"]
