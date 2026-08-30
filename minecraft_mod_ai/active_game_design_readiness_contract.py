from __future__ import annotations

"""Readiness contract for the current host-owned ``GameDesignPlanner`` path.

The modern planner deliberately avoids a model repair loop and merges one constrained
model response into a host skeleton.  That is safe only if empty skeleton fields cannot
be mistaken for a completed design.  This contract keeps the one-call policy, strengthens
the response schema, preserves requirement trace fields through normalization, and fails
closed before pre-retrieval planning when the frozen authored graph is not implemented by
the merged design.
"""

import json
from collections.abc import Mapping
from functools import wraps
from typing import Any

from . import evidence_request_guard as _request_guard
from . import game_design as _game_design
from .planner_design_readiness_contract import _validate_design_coverage
from .spec import SpecValidationError

_INSTALLED = False


def _active_authority() -> tuple[str, tuple[dict[str, Any], ...]] | None:
    active = _request_guard._ACTIVE_REQUEST_CATALOG.get()
    if active is None:
        return None
    prompt, catalog = active
    raw_requirements = catalog.get("requirements", [])
    if not isinstance(raw_requirements, list) or not raw_requirements:
        return None
    ledger: list[dict[str, Any]] = []
    for raw in raw_requirements:
        if not isinstance(raw, Mapping):
            continue
        requirement_id = str(raw.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        span = raw.get("source_span")
        authored_text = (
            str(span.get("text") or "").strip()
            if isinstance(span, Mapping)
            else str(raw.get("statement") or "").strip()
        )
        ledger.append(
            {
                "requirement_id": requirement_id,
                "capability": str(raw.get("capability") or "").strip(),
                "authored_text": authored_text,
                "semantic_statement": str(raw.get("semantic_statement") or "").strip(),
                "acceptance": list(raw.get("acceptance") or []),
            }
        )
    return (prompt, tuple(ledger)) if ledger else None


def _install_response_schema() -> None:
    module_schema = {
        "type": "object",
        "properties": {
            "plugin_id": {"type": "string", "minLength": 1},
            "status": {"type": "string", "minLength": 1},
            "reason": {"type": "string", "minLength": 1},
            "requirement_refs": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "implementation_obligations": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": [
            "plugin_id",
            "status",
            "reason",
            "requirement_refs",
            "implementation_obligations",
        ],
        "additionalProperties": False,
    }
    asset_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "kind": {"type": "string", "minLength": 1},
            "brief": {"type": "string", "minLength": 1},
        },
        "required": ["id", "kind", "brief"],
        "additionalProperties": False,
    }
    design_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "pitch": {"type": "string", "minLength": 1},
            "core_loop": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "progression": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "combat": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "mod_context": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "modules": {
                "type": "array",
                "minItems": 1,
                "items": module_schema,
            },
            "assets": {"type": "array", "items": asset_schema},
            "acceptance_tests": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "art_direction": {"type": "object"},
        },
        "required": list(_game_design._GAME_DESIGN_FIELDS),
        "additionalProperties": False,
    }
    _game_design._GAME_DESIGN_RESPONSE_SCHEMA["properties"]["game_design"] = design_schema


def _install_trace_preserving_module_normalizer() -> None:
    original = _game_design._modules
    if getattr(original, "__mmm_requirement_trace_preserving__", False):
        return

    @wraps(original)
    def traced(value: Any):
        normalized = original(value)
        if not isinstance(value, list) or not normalized:
            return normalized
        raw_by_id: dict[str, Mapping[str, Any]] = {}
        for raw in value:
            if not isinstance(raw, Mapping):
                continue
            module_id = _game_design._identifier(
                raw.get("plugin_id") or raw.get("module_id") or raw.get("id")
            )
            if module_id and module_id not in raw_by_id:
                raw_by_id[module_id] = raw
        output: list[dict[str, Any]] = []
        for item in normalized:
            row: dict[str, Any] = dict(item)
            raw = raw_by_id.get(str(item.get("plugin_id") or ""))
            if raw is not None:
                row["requirement_refs"] = _game_design._strings(
                    raw.get("requirement_refs")
                )
                row["implementation_obligations"] = _game_design._strings(
                    raw.get("implementation_obligations")
                )
            output.append(row)
        return output

    traced.__mmm_requirement_trace_preserving__ = True  # type: ignore[attr-defined]
    _game_design._modules = traced


def _assert_minimum_design_depth(design: Mapping[str, Any]) -> None:
    for field in ("core_loop", "progression", "acceptance_tests"):
        value = design.get(field)
        if not isinstance(value, list) or not any(str(item).strip() for item in value):
            raise SpecValidationError(
                f"design readiness failed: {field} is empty after model normalization"
            )
    authority = _active_authority()
    if authority is not None:
        modules = design.get("modules")
        if not isinstance(modules, list) or not modules:
            raise SpecValidationError(
                "design readiness failed: modules are empty after model normalization"
            )


def _augment_system_prompt(system_prompt: str, ledger: tuple[Mapping[str, Any], ...]) -> str:
    compact = [
        {
            "requirement_id": item["requirement_id"],
            "capability": item.get("capability", ""),
            "authored_text": item.get("authored_text", ""),
            "semantic_statement": item.get("semantic_statement", ""),
            "acceptance": item.get("acceptance", []),
        }
        for item in ledger
    ]
    return system_prompt + (
        "\n\nFROZEN REQUIREMENT AUTHORITY (host-owned; do not rewrite IDs):\n"
        + json.dumps(compact, ensure_ascii=False, sort_keys=True)
        + "\nEvery approved requirement must appear in at least one modules[].requirement_refs. "
        "Each such module must contain concrete implementation_obligations. Preserve the "
        "authored behavior; do not invent target versions, mappings, API signatures, or "
        "unrequested mechanics. All required game-design fields must be substantively filled."
    )


def _install_generation_gate() -> None:
    original = _game_design._generate_game_design_once
    if getattr(original, "__mmm_active_design_readiness__", False):
        return

    @wraps(original)
    def guarded(*args: Any, **kwargs: Any):
        authority = _active_authority()
        call_kwargs = dict(kwargs)
        if authority is not None:
            full_prompt, ledger = authority
            call_kwargs["system_prompt"] = _augment_system_prompt(
                str(call_kwargs.get("system_prompt") or ""),
                ledger,
            )
        result = original(*args, **call_kwargs)
        if not isinstance(result, Mapping):
            raise SpecValidationError("game design generation returned a non-object result")
        _assert_minimum_design_depth(result)
        if authority is None:
            return result
        full_prompt, ledger = authority
        authoritative_prompt = str(call_kwargs.get("authoritative_prompt") or "")
        if authoritative_prompt == full_prompt:
            return _validate_design_coverage(result, ledger)
        return result

    guarded.__mmm_active_design_readiness__ = True  # type: ignore[attr-defined]
    _game_design._generate_game_design_once = guarded


def _install_sharded_merge_gate() -> None:
    original = _game_design._merge_game_design_pages
    if getattr(original, "__mmm_sharded_design_readiness__", False):
        return

    @wraps(original)
    def guarded(pages: Any):
        result = original(pages)
        if not isinstance(result, Mapping):
            raise SpecValidationError("sharded game design merge returned a non-object result")
        _assert_minimum_design_depth(result)
        authority = _active_authority()
        if authority is None:
            return result
        _prompt, ledger = authority
        return _validate_design_coverage(result, ledger)

    guarded.__mmm_sharded_design_readiness__ = True  # type: ignore[attr-defined]
    _game_design._merge_game_design_pages = guarded


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_response_schema()
    _install_trace_preserving_module_normalizer()
    _install_generation_gate()
    _install_sharded_merge_gate()
    _INSTALLED = True


__all__ = ["install"]
