from __future__ import annotations

"""Acyclic loader for authored execution concern contracts.

The canonical YAML files remain the single source of truth. This module deliberately
depends only on stdlib + yaml/jsonschema so authored execution schema loading cannot
pull the planning/runtime package graph back into itself.
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator

_ROOT = Path(__file__).with_name("templates").resolve()


def _canonical_id(identifier: str) -> str:
    value = str(identifier or "").strip()
    parsed = PurePosixPath(value)
    if (
        not value
        or parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or "\\" in value
    ):
        raise ValueError(f"AUTHORED_CONCERN_PATH: {identifier!r}")
    return value


@lru_cache(maxsize=None)
def _load_yaml(identifier: str) -> dict[str, Any]:
    canonical = _canonical_id(identifier)
    path = (_ROOT / f"{canonical}.yaml").resolve()
    if not path.is_relative_to(_ROOT) or not path.is_file():
        raise ValueError(f"AUTHORED_CONCERN_MISSING: {canonical}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("id") != canonical:
        raise ValueError(f"AUTHORED_CONCERN_ID: {canonical}")
    return deepcopy(value)


def _code_rules(raw: Any) -> list[str]:
    """Keep semantic/evidence rules; drop record-loop/output-control instructions."""
    result: list[str] = []
    for item in raw if isinstance(raw, list) else ():
        text = str(item).strip()
        lowered = text.casefold()
        if not text:
            continue
        if lowered.startswith((
            "return only ",
            "return done",
            "return not_applicable",
            "return blocked",
            "do not turn missing information into an inapplicability claim",
        )):
            continue
        if text not in result:
            result.append(text)
    return result


@lru_cache(maxsize=None)
def _compiled_section(section: str) -> tuple[dict[str, Any], ...]:
    name = str(section or "").strip()
    manifest = _load_yaml(f"criterion/{name}")
    steps = manifest.get("steps")
    if manifest.get("execution") != "sequence" or not isinstance(steps, list) or not steps:
        raise ValueError(f"AUTHORED_EXECUTION_SCHEMA: invalid criterion/{name}")
    contracts: list[dict[str, Any]] = []
    for sequence, raw_identifier in enumerate(steps):
        identifier = _canonical_id(str(raw_identifier))
        template = _load_yaml(identifier)
        task = str(template.get("task") or "").strip()
        schema = template.get("record_schema")
        if not task or not isinstance(schema, dict):
            raise ValueError(f"AUTHORED_CONCERN_CONTRACT: {identifier}")
        Draft202012Validator.check_schema(schema)
        contracts.append({
            "sequence": sequence,
            "identifier": identifier,
            "concern": identifier.rsplit("/", 1)[-1],
            "task": task,
            "rules": _code_rules(template.get("rules")),
            "record_schema": deepcopy(schema),
        })
    return tuple(contracts)


def load_authored_concern_contracts(section: str) -> tuple[dict[str, Any], ...]:
    return tuple(deepcopy(item) for item in _compiled_section(section))


__all__ = ["load_authored_concern_contracts"]
