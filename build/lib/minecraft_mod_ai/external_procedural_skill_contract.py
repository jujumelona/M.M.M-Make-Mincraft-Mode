from __future__ import annotations

"""Evidence-grounded procedural SkillBank utilities.

This module is deliberately not a runtime patch installer. Research notes may contain
procedures, and the canonical pre-design pipeline explicitly compiles those procedures
into a request/workspace SkillBank. Declarative research remains the evidence authority;
procedures never authorize tools or certify correctness.
"""

import hashlib
import json
import os
import stat
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, tuple[threading.RLock, int]] = {}
_SCHEMA_VERSION = "mmm/external-procedural-skillbank-v1"
_MAX_STEPS = 8
_MAX_ITEMS = 6
_MAX_SKILLS = 256


@contextmanager
def _skillbank_lock(path: Path):
    key = str(path.expanduser().resolve())
    with _PATH_LOCKS_GUARD:
        lock, users = _PATH_LOCKS.get(key, (threading.RLock(), 0))
        _PATH_LOCKS[key] = (lock, users + 1)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _PATH_LOCKS_GUARD:
            current = _PATH_LOCKS.get(key)
            if current is not None and current[0] is lock:
                remaining = current[1] - 1
                if remaining <= 0:
                    _PATH_LOCKS.pop(key, None)
                else:
                    _PATH_LOCKS[key] = (lock, remaining)


def _bounded_strings(value: Any, *, limit: int, chars: int = 320) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value[:limit]:
        text = " ".join(str(item).split())[:chars]
        if text and text not in result:
            result.append(text)
    return result


def _procedure_schema() -> dict[str, Any]:
    """Schema for evidence-supported reusable procedures and explicit dependencies."""

    return {
        "type": "array",
        "maxItems": _MAX_ITEMS,
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 160},
                "activate_when": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {"type": "string", "minLength": 1, "maxLength": 320},
                },
                "contraindications": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {"type": "string", "minLength": 1, "maxLength": 320},
                },
                "steps": {
                    "type": "array",
                    "maxItems": _MAX_STEPS,
                    "items": {"type": "string", "minLength": 1, "maxLength": 420},
                },
                "constraints": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {"type": "string", "minLength": 1, "maxLength": 320},
                },
                "output_contract": {"type": "string", "maxLength": 420},
                "evidence_refs": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {"type": "string", "minLength": 1, "maxLength": 192},
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "requires": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 192},
                },
                "provides": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 192},
                },
            },
            "required": [
                "name",
                "activate_when",
                "contraindications",
                "steps",
                "constraints",
                "output_contract",
                "evidence_refs",
                "confidence",
                "requires",
                "provides",
            ],
            "additionalProperties": False,
        },
    }


def _sanitize_procedure(
    value: Mapping[str, Any],
    domain_id: str,
) -> dict[str, Any] | None:
    """Accept only compact procedures with cited evidence and explicit dependency labels."""

    name = " ".join(str(value.get("name", "")).split())[:160]
    activate_when = _bounded_strings(value.get("activate_when"), limit=4)
    contraindications = _bounded_strings(value.get("contraindications"), limit=4)
    steps = _bounded_strings(value.get("steps"), limit=_MAX_STEPS, chars=420)
    constraints = _bounded_strings(value.get("constraints"), limit=6, chars=320)
    evidence_refs = _bounded_strings(value.get("evidence_refs"), limit=6, chars=192)
    requires = _bounded_strings(value.get("requires"), limit=8, chars=192)
    provides = _bounded_strings(value.get("provides"), limit=8, chars=192)
    output_contract = " ".join(str(value.get("output_contract", "")).split())[:420]
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if not name or not activate_when or not steps or not evidence_refs:
        return None

    canonical = {
        "domain_id": domain_id,
        "name": name,
        "activate_when": activate_when,
        "contraindications": contraindications,
        "steps": steps,
        "constraints": constraints,
        "output_contract": output_contract,
        "evidence_refs": evidence_refs,
        "confidence": round(confidence, 4),
        "requires": requires,
        "provides": provides,
        "rule": (
            "Procedural guidance compiled from cited research evidence. Re-check current "
            "preconditions and exact-version evidence; compiler, tests, runtime observations "
            "and host validators override this skill."
        ),
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    canonical["skill_id"] = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return canonical


def _compile_skillbank(domain_notes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compile only procedures explicitly present in evidence-grounded domain notes."""

    skills: list[dict[str, Any]] = []
    seen: set[str] = set()
    for note in domain_notes:
        domain_id = str(note.get("domain_id", "")).strip() or "unknown"
        raw = note.get("procedures")
        if not isinstance(raw, list):
            continue
        for item in raw[:_MAX_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            skill = _sanitize_procedure(item, domain_id)
            if skill is None:
                continue
            identity = str(skill["skill_id"])
            if identity in seen:
                continue
            seen.add(identity)
            skills.append(skill)

    skills.sort(
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            str(item.get("domain_id", "")),
            str(item.get("skill_id", "")),
        )
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "skills": skills,
        "source_skill_count": len(skills),
        "relation_graph": [],
        "policy": (
            "Declarative RAG remains evidence authority. Skills are reusable procedures "
            "only and never authorize tools or certify correctness."
        ),
    }


def _skillbank_path(router: Any) -> Path | None:
    root = getattr(router, "_agent_workspace_root", None)
    if root is None:
        return None
    workspace = Path(root).expanduser().resolve()
    metadata = workspace / ".minecraft_ai"
    try:
        if metadata.exists() and metadata.is_symlink():
            return None
    except OSError:
        return None
    return metadata / "procedural-skillbank.jsonl"


def _load_persistent_skills(
    path: Path,
    *,
    limit: int = _MAX_SKILLS,
) -> list[dict[str, Any]]:
    try:
        info = path.lstat()
    except (FileNotFoundError, OSError):
        return []
    if not stat.S_ISREG(info.st_mode):
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(value, dict)
                    and value.get("schema_version") == _SCHEMA_VERSION
                    and value.get("skill_id")
                ):
                    rows.append(value)
    except OSError:
        return []
    return rows[-max(1, int(limit)) :]


def _persist_skills(
    path: Path,
    skills: Sequence[Mapping[str, Any]],
    *,
    prior: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if not skills:
        return
    with _skillbank_lock(path):
        if path.exists() and (path.is_symlink() or not path.is_file()):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = prior if prior is not None else _load_persistent_skills(path)
        by_id = {
            str(row.get("skill_id", "")): dict(row)
            for row in existing
            if str(row.get("skill_id", ""))
        }
        for skill in skills:
            identity = str(skill.get("skill_id", ""))
            if identity:
                row = dict(skill)
                row["schema_version"] = _SCHEMA_VERSION
                by_id[identity] = row
        ordered = sorted(
            by_id.values(),
            key=lambda row: (
                -float(row.get("confidence", 0.0) or 0.0),
                str(row.get("skill_id", "")),
            ),
        )[:_MAX_SKILLS]
        temp = path.with_suffix(path.suffix + ".tmp")
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as handle:
                for row in ordered:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(temp, path)
        except OSError:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


def _lexical_terms(value: str) -> set[str]:
    return {term.casefold() for term in value.split() if term.strip()}


def _select_skills(
    query: str,
    skills: Sequence[Mapping[str, Any]],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Rank only skills sharing authored terms; no semantic similarity cutoff is used."""

    target = _lexical_terms(query)
    ranked: list[tuple[int, float, str, dict[str, Any]]] = []
    for skill in skills:
        document = " ".join(
            [
                str(skill.get("name", "")),
                *[str(item) for item in skill.get("activate_when", ())],
                *[str(item) for item in skill.get("steps", ())],
                *[str(item) for item in skill.get("constraints", ())],
            ]
        )
        overlap = len(target & _lexical_terms(document))
        if target and overlap == 0:
            continue
        try:
            confidence = float(skill.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        ranked.append(
            (
                overlap,
                max(0.0, min(1.0, confidence)),
                str(skill.get("skill_id", "")),
                dict(skill),
            )
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [row for _overlap, _confidence, _identity, row in ranked[: max(1, int(limit))]]


def attach_procedural_skillbank(
    router: Any,
    prompt: str,
    research: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the current/retrieved SkillBank without mutating any research owner."""

    value = dict(research)
    raw_notes = value.get("domain_notes")
    notes = [item for item in raw_notes if isinstance(item, Mapping)] if isinstance(raw_notes, list) else []
    current_bank = _compile_skillbank(notes)

    path = _skillbank_path(router)
    prior: list[dict[str, Any]] = []
    if path is not None:
        with _skillbank_lock(path):
            prior = _load_persistent_skills(path)
            _persist_skills(path, current_bank["skills"], prior=prior)

    available = [*prior, *current_bank["skills"]]
    retrieved = _select_skills(prompt, available, limit=6)
    value["procedural_skillbank"] = {
        **current_bank,
        "retrieved_skills": retrieved,
        "persistence": (
            "workspace/.minecraft_ai/procedural-skillbank.jsonl"
            if path is not None
            else "request_scoped"
        ),
    }
    method = dict(value.get("method", {})) if isinstance(value.get("method"), Mapping) else {}
    method["procedural_rag"] = (
        "evidence-grounded procedures compiled explicitly from domain research notes"
    )
    value["method"] = method
    return value


def compact_skillbank(research: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the bounded model-facing SkillBank receipt."""

    bank = research.get("procedural_skillbank")
    if not isinstance(bank, Mapping):
        return None
    result: dict[str, Any] = {
        "schema_version": bank.get("schema_version"),
        "retrieved_skills": list(bank.get("retrieved_skills", ()))[:6],
        "current_skills": list(bank.get("skills", ()))[:8],
        "relation_graph": list(bank.get("relation_graph", ()))[:12],
        "policy": bank.get("policy"),
    }
    composition = bank.get("skill_composition")
    if isinstance(composition, Mapping):
        result["skill_composition"] = {
            "schema_version": composition.get("schema_version"),
            "composition_policy": composition.get("composition_policy"),
            "ordered_skills": list(composition.get("ordered_skills", ()))[:12],
            "dependency_edges": list(composition.get("dependency_edges", ()))[:24],
            "unresolved_requirements": list(composition.get("unresolved_requirements", ()))[:12],
            "cycles": list(composition.get("cycles", ()))[:8],
            "blocked_skill_ids": list(composition.get("blocked_skill_ids", ()))[:16],
        }
    return result


__all__ = [
    "attach_procedural_skillbank",
    "compact_skillbank",
]
