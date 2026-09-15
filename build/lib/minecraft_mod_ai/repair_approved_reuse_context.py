from __future__ import annotations

"""Verified donor-source handoff for the repair coder.

Repair is allowed to reuse source only when the approved complete proposal already
contains a source-transplant/adaptation decision. The host replays that verified plan
through ``materialize_source_slices`` before exposing any donor bytes, so ordinary RAG
references can never become source-reuse authority merely by looking similar.
"""

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

from .production_tools import ProductionToolService
from .proposal_store import read_sharded_complete_proposal_section
from .source_transplant import SourceTransplantError, materialize_source_slices
from .spec import SpecValidationError

_REPAIR_REUSE_SCHEMA = "mmm/approved-repair-reuse-context-v1"
_DEFAULT_REPAIR_REUSE_BYTES = 12 * 1024
_MAX_REUSE_PLAN_SECTION_BYTES = 4 * 1024 * 1024
_TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]{2,}")
_MARKER = "_mmm_approved_repair_reuse_handoff_v1"


class RepairReuseContextError(RuntimeError):
    """Approved donor state existed but could not be verified for repair use."""


def _sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_donor_decisions(
    plan: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(plan, Mapping):
        return ()
    raw = plan.get("capabilities")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(
        item
        for item in raw
        if isinstance(item, Mapping)
        and str(item.get("mode") or "").strip().casefold()
        in {"source_transplant", "adapt"}
        and isinstance(item.get("donor"), Mapping)
    )


def _load_approved_reuse_plan(root: Path) -> dict[str, Any] | None:
    """Read only the persisted reuse-plan field instead of loading the full proposal."""

    index = root / ".minecraft_ai" / "complete-proposal.json"
    if not index.is_file() or index.is_symlink():
        return None
    try:
        page = read_sharded_complete_proposal_section(
            index,
            "game_design._reuse_plan",
            limit=1000,
            max_bytes=_MAX_REUSE_PLAN_SECTION_BYTES,
        )
    except SpecValidationError as exc:
        if "Unknown complete proposal section" in str(exc):
            return None
        raise RepairReuseContextError(
            f"Persisted approved reuse plan is invalid: {exc}"
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairReuseContextError(
            f"Persisted approved reuse plan could not be read: {type(exc).__name__}: {exc}"
        ) from exc

    if page.get("item_fragment") is not None or page.get("next_cursor"):
        raise RepairReuseContextError(
            "Persisted approved reuse plan exceeds the bounded repair handoff section."
        )
    items = page.get("items")
    if not isinstance(items, list):
        raise RepairReuseContextError("Persisted approved reuse plan page has no items.")
    plan: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise RepairReuseContextError("Persisted approved reuse plan item is malformed.")
        key = item.get("key")
        if not isinstance(key, str) or not key or key in plan or "value" not in item:
            raise RepairReuseContextError("Persisted approved reuse plan key is malformed.")
        plan[key] = item["value"]
    total_count = page.get("total_count")
    if type(total_count) is not int or total_count != len(plan):
        raise RepairReuseContextError(
            "Persisted approved reuse plan was not read as one complete bounded object."
        )
    return plan


def _diagnostic_terms(diagnostic: Mapping[str, Any]) -> set[str]:
    terms: set[str] = set()
    for key in ("symbols", "exceptions", "files", "tasks", "messages"):
        raw = diagnostic.get(key, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raw = (raw,) if raw else ()
        for value in raw:
            terms.update(token.casefold() for token in _TOKEN_RE.findall(str(value)))
            if key == "files" and str(value).strip():
                terms.add(Path(str(value)).name.casefold())
    return terms


def _candidate_score(
    donor: Mapping[str, Any],
    file_receipt: Mapping[str, Any],
    terms: set[str],
) -> tuple[int, str]:
    path = str(file_receipt.get("source_path") or file_receipt.get("path") or "")
    values = [
        path,
        str(donor.get("repository") or ""),
        str(donor.get("capability") or ""),
        *[str(value) for value in file_receipt.get("symbols", ()) if str(value)],
    ]
    candidate_terms = {
        token.casefold()
        for value in values
        for token in _TOKEN_RE.findall(value)
    }
    basename = Path(path).name.casefold() if path else ""
    overlap = len(candidate_terms & terms)
    path_bonus = 3 if basename and basename in terms else 0
    symbol_bonus = sum(
        2
        for value in file_receipt.get("symbols", ())
        if str(value).casefold() in terms
    )
    return overlap * 10 + path_bonus + symbol_bonus, path


def build_approved_repair_reuse_context(
    root: Path,
    diagnostic: Mapping[str, Any],
    *,
    byte_budget: int = _DEFAULT_REPAIR_REUSE_BYTES,
) -> dict[str, Any] | None:
    """Expose bounded donor code only after replaying the approved reuse proof path."""

    normalized_root = root.expanduser().resolve()
    plan = _load_approved_reuse_plan(normalized_root)
    decisions = _source_donor_decisions(plan)
    if not decisions:
        return None

    try:
        materialization = materialize_source_slices(normalized_root, plan or {})
    except (OSError, ValueError, SpecValidationError, SourceTransplantError) as exc:
        raise RepairReuseContextError(
            f"Approved repair donor verification failed: {type(exc).__name__}: {exc}"
        ) from exc

    donors = materialization.get("donors")
    if (
        not isinstance(donors, list)
        or materialization.get("count") != len(donors)
        or len(donors) != len(decisions)
    ):
        raise RepairReuseContextError(
            "Approved repair reuse plan did not materialize every verified donor."
        )

    terms = _diagnostic_terms(diagnostic)
    candidates: list[tuple[int, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for donor in donors:
        if not isinstance(donor, Mapping):
            raise RepairReuseContextError("Approved donor materialization receipt is malformed.")
        files = donor.get("files")
        if not isinstance(files, list):
            raise RepairReuseContextError("Approved donor materialization has no file list.")
        for file_receipt in files:
            if not isinstance(file_receipt, Mapping):
                continue
            score, path = _candidate_score(donor, file_receipt, terms)
            candidates.append((-score, path, donor, file_receipt))
    candidates.sort(key=lambda item: (item[0], item[1]))

    effective_budget = max(1024, int(byte_budget))
    remaining = effective_budget
    snippets: list[dict[str, Any]] = []
    service = ProductionToolService(workspace_root=normalized_root)
    try:
        for neg_score, _path, _donor, file_receipt in candidates:
            if remaining <= 0:
                break
            path = str(file_receipt.get("path") or "")
            if not path:
                continue
            try:
                source = service.read_reuse_source(
                    ".",
                    path,
                    limit_bytes=min(6 * 1024, remaining),
                )
            except (OSError, ValueError, SpecValidationError) as exc:
                raise RepairReuseContextError(
                    f"Approved repair donor source could not be read: {type(exc).__name__}: {exc}"
                ) from exc
            content = str(source.get("content") or "")
            used = len(content.encode("utf-8"))
            if not content or used <= 0:
                continue
            remaining = max(0, remaining - used)
            snippets.append(
                {
                    "repository": source.get("repository"),
                    "commit_sha": source.get("commit_sha"),
                    "license_id": source.get("license_id"),
                    "capability": source.get("capability"),
                    "path": source.get("path"),
                    "sha256": source.get("sha256"),
                    "symbols": list(file_receipt.get("symbols") or ()),
                    "relevance_score": max(0, -neg_score),
                    "content": content,
                }
            )
    finally:
        service.close()

    if not snippets:
        raise RepairReuseContextError(
            "Approved repair donors materialized without readable source snippets."
        )
    return {
        "schema_version": _REPAIR_REUSE_SCHEMA,
        "status": "APPROVED",
        "diagnostic_sha256": _sha(diagnostic),
        "materialization": materialization,
        "snippets": snippets,
        "byte_budget": effective_budget,
        "bytes_used": effective_budget - remaining,
        "policy": {
            "source_reuse_authority": "verified_reuse_plan_only",
            "reference_rag_never_authorizes_source_reuse": True,
            "repair_must_preserve_commit_license_provenance": True,
            "prefer_diagnostic_relevant_verified_donor_before_fresh_reimplementation": True,
        },
    }


def install(repair_module: Any) -> None:
    """Compose verified donor context after the canonical repair evidence wrapper."""

    cls = repair_module.RepairEngine
    current = cls._context
    if getattr(current, _MARKER, False):
        return

    from . import research_coder_repair_reuse as reuse_hardener

    @wraps(current)
    def context(self: Any, root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
        normalized_root = root.expanduser().resolve()
        base = dict(current(self, normalized_root, evidence))
        diagnostic = reuse_hardener._diagnostic_signature_payload(evidence)
        cache_key = f"{normalized_root}:{_sha(diagnostic)}"
        cache = getattr(self, "_mmm_approved_repair_reuse_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._mmm_approved_repair_reuse_cache = cache
        if cache_key in cache:
            approved_context = copy.deepcopy(cache[cache_key])
        else:
            try:
                approved_context = build_approved_repair_reuse_context(
                    normalized_root,
                    diagnostic,
                )
            except RepairReuseContextError as exc:
                approved_context = {
                    "schema_version": _REPAIR_REUSE_SCHEMA,
                    "status": "REJECTED",
                    "snippets": [],
                    "error": str(exc)[:1024],
                    "policy": {
                        "source_reuse_authority": "none",
                        "reference_rag_never_authorizes_source_reuse": True,
                        "repair_must_not_guess_or_reuse_unverified_source": True,
                    },
                }
            cache[cache_key] = copy.deepcopy(approved_context)
            while len(cache) > 16:
                cache.pop(next(iter(cache)))

        if approved_context is not None:
            base["approved_reuse_context"] = copy.deepcopy(approved_context)
        policy = dict(base.get("retrieval_policy") or {})
        policy.update(
            {
                "verified_donor_required_for_source_reuse": True,
                "reference_evidence_cannot_authorize_source_reuse": True,
                "approved_repair_donor_available": bool(
                    isinstance(approved_context, Mapping)
                    and approved_context.get("status") == "APPROVED"
                ),
            }
        )
        base["retrieval_policy"] = policy
        return base

    setattr(context, _MARKER, True)
    context.__wrapped__ = current  # type: ignore[attr-defined]
    cls._context = context


__all__ = [
    "RepairReuseContextError",
    "build_approved_repair_reuse_context",
    "install",
]
