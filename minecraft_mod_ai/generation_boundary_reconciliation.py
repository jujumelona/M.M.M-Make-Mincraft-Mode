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
_MODEL_PREAMBLE_META_RE = re.compile(
    r"(?:"
    r"<\s*/?\s*think\b"
    r"|^\s*(?:thinking\s+process|analysis|reasoning|internal\s+reasoning|"
    r"analyze\s+the\s+request|drafting\s+content|formatting\s+check|"
    r"final\s+review(?:\s+of\s+constraints)?|plan)\s*:"
    r"|\bi\s+need\s+to\b"
    r"|\bi\s+should\b"
    r"|\bthe\s+user\s+wants\b"
    r"|\bbranch[- ]policy\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_CONTENT_LABEL_RE = re.compile(
    r"^\s*Content(?:\s*[:\-–—]\s*|\s+(?=[^\x00-\x7F]))",
    re.IGNORECASE,
)
_IDENTITY_WRAPPERS = {
    "title": (
        re.compile(
            r"^\s*(?:here(?:'s| is)|below is)\s+(?:the\s+)?title\s*[:\-–—]\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:the\s+)?title\s*(?:is\s*[:\-–—]?|[:\-–—])\s*",
            re.IGNORECASE,
        ),
    ),
    "pitch": (
        re.compile(
            r"^\s*(?:here(?:'s| is)|below is)\s+(?:the\s+)?pitch\s*[:\-–—]\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:the\s+)?pitch\s*(?:is\s*[:\-–—]?|[:\-–—])\s*",
            re.IGNORECASE,
        ),
    ),
}


def _normalize_heading_for_boundary(value: str) -> str:
    value = value.strip().strip("`").casefold()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _sanitize_section_output(raw: Any, fields: Any) -> str:
    """Drop only a clearly model-meta preamble before the first approved section heading.

    The host-owned section parser already treats text before the first approved heading as
    non-field content. This function makes that boundary explicit for Qwen variants that
    emit ``Thinking Process:`` and an orphan closing ``</think>`` before their Markdown
    answer. Arbitrary prose is not stripped merely because it precedes a heading: at least
    one known internal-meta marker must be present. Once an approved heading begins, no
    meta content is hidden; the canonical field validator remains fail-closed.
    """

    text = str(raw or "")
    expected = {
        _normalize_heading_for_boundary(str(field))
        for field in fields
        if str(field).strip()
    }
    lines = text.splitlines()
    first_heading_index: int | None = None
    for index, line in enumerate(lines):
        match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if match and _normalize_heading_for_boundary(match.group(1)) in expected:
            first_heading_index = index
            break
    if first_heading_index is None or first_heading_index <= 0:
        return text
    preamble = "\n".join(lines[:first_heading_index])
    if not _MODEL_PREAMBLE_META_RE.search(preamble):
        return text
    return "\n".join(lines[first_heading_index:])


def _sanitize_design_body(raw: Any, field: str) -> str:
    """Remove bounded scaffolding while leaving semantic field validation unchanged.

    Balanced ``<think>...</think>`` blocks have explicit boundaries and can be removed
    without guessing. Unbalanced markers *inside an actual field* are deliberately kept so
    ``assert_design_field_clean`` rejects/falls back. Identity fields also accept a narrow
    set of model-added labels. The bare ``Content`` label is stripped only when followed
    by non-ASCII content (the exact Qwen artefact observed in Korean design output) or an
    explicit delimiter, avoiding corruption of legitimate English titles such as
    ``Content Warning``.
    """

    body = str(raw or "")
    body = _THINK_BLOCK_RE.sub("", body)
    if field in _IDENTITY_WRAPPERS:
        stripped = body.strip()
        stripped = _CONTENT_LABEL_RE.sub("", stripped, count=1).strip()
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

    # Keep the canonical field parser/validator strict and independently testable. Cleanup
    # belongs at the model-output extraction boundary immediately before that parser runs.
    original_section_field_body = agentic._section_field_body
    if not getattr(original_section_field_body, "_mmm_meta_sanitized_boundary", False):

        @wraps(original_section_field_body)
        def section_field_body(raw: Any, field: str, fields: Any) -> str:
            sanitized_raw = _sanitize_section_output(raw, fields)
            body = original_section_field_body(sanitized_raw, field, fields)
            return _sanitize_design_body(body, field)

        section_field_body._mmm_meta_sanitized_boundary = True
        section_field_body.__wrapped__ = original_section_field_body
        agentic._section_field_body = section_field_body

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
