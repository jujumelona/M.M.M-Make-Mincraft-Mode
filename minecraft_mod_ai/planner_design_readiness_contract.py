from __future__ import annotations

"""Fail-closed requirement-to-design readiness for Worker 03.

The section planner already owns bounded generation and repair.  This contract does not
add a competing generator.  It strengthens the live owner's schema/messages/validator so
missing model output cannot be silently converted into an accepted empty design, and it
requires the frozen authored requirement graph to reach implementation-bearing design
modules before planning may continue.
"""

import json
import re
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from functools import wraps
from typing import Any

from . import agentic_research_game_design as _design
from . import evidence_first_planning as _evidence
from . import evidence_request_guard as _request_guard
from .spec import SpecValidationError

_INSTALLED = False
_ACTIVE_REQUIRED_IDS: ContextVar[tuple[str, ...]] = ContextVar(
    "mmm_design_required_ids",
    default=(),
)


def _active_requirement_ledger(prompt: str) -> tuple[dict[str, Any], ...]:
    """Read only the already-frozen request authority; never rebuild scope here."""

    active = _request_guard._ACTIVE_REQUEST_CATALOG.get()
    if active is None or active[0] != prompt:
        return ()
    catalog = active[1]
    raw_requirements = catalog.get("requirements", [])
    if not isinstance(raw_requirements, list):
        return ()
    ledger: list[dict[str, Any]] = []
    for raw in raw_requirements:
        if not isinstance(raw, Mapping):
            continue
        requirement_id = str(raw.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        span = raw.get("source_span")
        span_text = (
            str(span.get("text") or "").strip()
            if isinstance(span, Mapping)
            else ""
        )
        behavior = raw.get("observable_behavior")
        acceptance = raw.get("acceptance")
        ledger.append(
            {
                "requirement_id": requirement_id,
                "capability": str(raw.get("capability") or "").strip(),
                "authored_text": span_text or str(raw.get("statement") or "").strip(),
                "semantic_statement": str(raw.get("semantic_statement") or "").strip(),
                "observable_behavior": dict(behavior) if isinstance(behavior, Mapping) else {},
                "acceptance": [
                    str(item).strip()
                    for item in acceptance
                    if str(item).strip()
                ]
                if isinstance(acceptance, list)
                else [],
            }
        )
    return tuple(ledger)


def _install_module_trace_schema() -> None:
    for section_id, _fields, properties in _design._SECTION_SPECS:
        if section_id != "modules_and_assets":
            continue
        modules = properties.get("modules")
        if not isinstance(modules, dict):
            raise RuntimeError("modules_and_assets schema lost its modules object")
        item_schema = modules.get("items")
        if not isinstance(item_schema, dict):
            raise RuntimeError("modules schema lost its item object")
        item_properties = item_schema.get("properties")
        required = item_schema.get("required")
        if not isinstance(item_properties, dict) or not isinstance(required, list):
            raise RuntimeError("modules schema is not contract-compatible")
        item_properties.setdefault(
            "requirement_refs",
            {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        )
        item_properties.setdefault(
            "implementation_obligations",
            {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        )
        for field in ("requirement_refs", "implementation_obligations"):
            if field not in required:
                required.append(field)
        return
    raise RuntimeError("modules_and_assets section schema was not found")


def _nonempty_text_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SpecValidationError(f"{field} must be a non-empty list; empty accepted design is forbidden")
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if len(cleaned) != len(value) or not cleaned:
        raise SpecValidationError(f"{field} must contain only non-empty authored design entries")
    return cleaned


def _strict_validate_section_types(
    section: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    """Validate model output without host-authored semantic fallback/coercion."""

    for field in fields:
        if field not in section:
            raise SpecValidationError(f"section omitted required field {field!r}")
        value = section.get(field)
        if field in {"title", "pitch"}:
            text = str(value).strip() if isinstance(value, str) else ""
            if not text or text == f"Generated {field}":
                raise SpecValidationError(
                    f"{field} must be supplied by the planner; host fallback is not accepted"
                )
        elif field in {"core_loop", "progression", "acceptance_tests"}:
            _nonempty_text_list(value, field=field)
        elif field == "assets":
            if not isinstance(value, list):
                raise SpecValidationError("assets must be a list")
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise SpecValidationError(f"assets[{index}] must be an object")
                for key in ("id", "kind", "brief"):
                    if not str(item.get(key) or "").strip():
                        raise SpecValidationError(f"assets[{index}].{key} must be non-empty")
        elif field == "modules":
            if not isinstance(value, list):
                raise SpecValidationError("modules must be a list")
            required_ids = set(_ACTIVE_REQUIRED_IDS.get())
            if required_ids and not value:
                raise SpecValidationError(
                    "modules must be non-empty while approved authored requirements exist"
                )
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise SpecValidationError(f"modules[{index}] must be an object")
                for key in ("plugin_id", "status", "reason"):
                    if not str(item.get(key) or "").strip():
                        raise SpecValidationError(f"modules[{index}].{key} must be non-empty")
                if required_ids:
                    refs = _nonempty_text_list(
                        item.get("requirement_refs"),
                        field=f"modules[{index}].requirement_refs",
                    )
                    obligations = _nonempty_text_list(
                        item.get("implementation_obligations"),
                        field=f"modules[{index}].implementation_obligations",
                    )
                    unknown = sorted(set(refs) - required_ids)
                    if unknown:
                        raise SpecValidationError(
                            "module cites unknown requirement ids: " + ", ".join(unknown)
                        )
                    if not obligations:
                        raise SpecValidationError(
                            f"modules[{index}] has no implementation obligations"
                        )
        elif field in {"combat", "mod_context", "art_direction"}:
            if not isinstance(value, dict):
                raise SpecValidationError(f"{field} must be an object")
            if field in {"combat", "mod_context"}:
                for key, items in value.items():
                    if not str(key).strip():
                        raise SpecValidationError(f"{field} contains an empty key")
                    _nonempty_text_list(items, field=f"{field}.{key}")


def _install_section_context() -> None:
    original = _design._generate_section
    if getattr(original, "__mmm_requirement_design_context__", False):
        return

    @wraps(original)
    def guarded(*args: Any, **kwargs: Any):
        prompt = str(kwargs.get("prompt") or "")
        ledger = _active_requirement_ledger(prompt)
        token = _ACTIVE_REQUIRED_IDS.set(
            tuple(item["requirement_id"] for item in ledger)
        )
        try:
            return original(*args, **kwargs)
        finally:
            _ACTIVE_REQUIRED_IDS.reset(token)

    guarded.__mmm_requirement_design_context__ = True  # type: ignore[attr-defined]
    _design._generate_section = guarded


def _install_requirement_messages() -> None:
    original = _design._section_messages
    if getattr(original, "__mmm_requirement_design_messages__", False):
        return

    @wraps(original)
    def requirement_messages(*args: Any, **kwargs: Any):
        messages = original(*args, **kwargs)
        prompt = str(kwargs.get("prompt") or "")
        ledger = _active_requirement_ledger(prompt)
        if not ledger:
            return messages
        rewritten = [dict(message) for message in messages]
        system = str(rewritten[0].get("content") or "")
        rewritten[0]["content"] = system + (
            " The approved_requirements ledger is frozen host authority. Preserve every "
            "requirement_id exactly. For modules, requirement_refs must cite only those IDs "
            "and implementation_obligations must state concrete work needed to realize them. "
            "Do not merge away, summarize away, or invent authored requirements."
        )
        payload = json.loads(str(rewritten[1].get("content") or "{}"))
        payload["approved_requirements"] = list(ledger)
        payload["requirement_traceability_instruction"] = (
            "Every approved requirement must be covered by at least one module with exact "
            "requirement_refs and non-empty implementation_obligations before design acceptance."
        )
        rewritten[1]["content"] = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        return rewritten

    requirement_messages.__mmm_requirement_design_messages__ = True  # type: ignore[attr-defined]
    _design._section_messages = requirement_messages


def _validate_design_coverage(
    design: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_ids = tuple(str(item["requirement_id"]) for item in ledger)
    if not required_ids:
        return dict(design)
    known = set(required_ids)
    modules = design.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SpecValidationError(
            "design readiness failed: approved requirements exist but modules are empty"
        )
    covered: set[str] = set()
    binding_rows: dict[str, dict[str, Any]] = {
        requirement_id: {
            "requirement_id": requirement_id,
            "module_ids": [],
            "implementation_obligations": [],
        }
        for requirement_id in required_ids
    }
    for index, item in enumerate(modules):
        if not isinstance(item, Mapping):
            raise SpecValidationError(f"modules[{index}] must be an object")
        module_id = str(item.get("plugin_id") or "").strip()
        refs = _nonempty_text_list(
            item.get("requirement_refs"),
            field=f"modules[{index}].requirement_refs",
        )
        obligations = _nonempty_text_list(
            item.get("implementation_obligations"),
            field=f"modules[{index}].implementation_obligations",
        )
        unknown = sorted(set(refs) - known)
        if unknown:
            raise SpecValidationError(
                "design readiness failed: unknown requirement refs " + ", ".join(unknown)
            )
        for requirement_id in refs:
            covered.add(requirement_id)
            row = binding_rows[requirement_id]
            if module_id not in row["module_ids"]:
                row["module_ids"].append(module_id)
            for obligation in obligations:
                if obligation not in row["implementation_obligations"]:
                    row["implementation_obligations"].append(obligation)
    missing = [requirement_id for requirement_id in required_ids if requirement_id not in covered]
    if missing:
        raise SpecValidationError(
            "design readiness failed: approved requirements have no implementation-bearing "
            "design module: " + ", ".join(missing)
        )
    result = dict(design)
    result["_requirement_design_bindings"] = {
        "schema_version": "mmm/requirement-design-binding-v1",
        "requirement_ids": list(required_ids),
        "bindings": [binding_rows[requirement_id] for requirement_id in required_ids],
    }
    return result


def _install_design_coverage_gate() -> None:
    original = _design.generate_sectioned_game_design
    if getattr(original, "__mmm_requirement_design_coverage__", False):
        return

    @wraps(original)
    def guarded(game_design_module: Any, router: Any, prompt: str, *args: Any, **kwargs: Any):
        ledger = _active_requirement_ledger(prompt)
        result = original(game_design_module, router, prompt, *args, **kwargs)
        if not isinstance(result, Mapping):
            raise SpecValidationError("sectioned game design returned a non-object result")
        return _validate_design_coverage(result, ledger)

    guarded.__mmm_requirement_design_coverage__ = True  # type: ignore[attr-defined]
    _design.generate_sectioned_game_design = guarded


def _lossless_semantic_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    """Semantic line spans whose offsets remain exact for LF, CRLF, and CR."""

    spans: list[tuple[int, int]] = []
    line_offset = 0
    for raw_with_eol in prompt.splitlines(keepends=True):
        raw_line = raw_with_eol.rstrip("\r\n")
        line_start = line_offset
        line_offset += len(raw_with_eol)
        stripped = re.sub(r"^[\s\-\*•▶●]+|^\s*\d+\.\s*", "", raw_line)
        if not stripped.strip():
            continue
        matched_any = False
        for match in _evidence._SEMANTIC_BOUNDARY.finditer(raw_line):
            start = line_start + match.start()
            end = line_start + match.end()
            inner = raw_line[match.start() : match.end()]
            inner_stripped = re.sub(
                r"^[\s\-\*•▶●]+|^\s*\d+\.\s*",
                "",
                inner,
            )
            if not inner_stripped.strip():
                continue
            leading = len(inner) - len(inner.lstrip())
            bullet_stripped = inner.lstrip()
            bullet_cleaned = re.sub(
                r"^[\-\*•▶●]+|^\d+\.\s*",
                "",
                bullet_stripped,
            )
            start += leading + (len(bullet_stripped) - len(bullet_cleaned))
            while start < end and prompt[start].isspace():
                start += 1
            while end > start and prompt[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
                matched_any = True
        if not matched_any:
            leading = len(raw_line) - len(raw_line.lstrip())
            bullet_stripped = raw_line.lstrip()
            bullet_cleaned = re.sub(
                r"^[\-\*•▶●]+|^\d+\.\s*",
                "",
                bullet_stripped,
            )
            start = line_start + leading + (len(bullet_stripped) - len(bullet_cleaned))
            end = line_start + len(raw_line.rstrip())
            while start < end and prompt[start].isspace():
                start += 1
            if start < end:
                spans.append((start, end))
    if not spans and prompt.strip():
        spans.append((len(prompt) - len(prompt.lstrip()), len(prompt.rstrip())))
    return tuple(spans)


def _install_lossless_source_offsets() -> None:
    if getattr(_evidence._semantic_spans, "__mmm_crlf_lossless__", False):
        return
    _lossless_semantic_spans.__mmm_crlf_lossless__ = True  # type: ignore[attr-defined]
    _evidence._semantic_spans = _lossless_semantic_spans


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_module_trace_schema()
    _design._validate_section_types = _strict_validate_section_types
    _install_requirement_messages()
    _install_section_context()
    _install_design_coverage_gate()
    _install_lossless_source_offsets()
    _INSTALLED = True


__all__ = ["install"]
