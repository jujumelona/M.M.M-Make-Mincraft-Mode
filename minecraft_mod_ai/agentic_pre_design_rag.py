from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .knowledge import (
    AuthoritativeEvidenceRetriever,
    evidence_catalog_for_version,
    target_neutral_evidence_catalog,
)
from .rag_index import ProjectRAGIndex

_EVIDENCE_PAGE_CHARS = 1_800
_EVIDENCE_DOCUMENT_SCHEMA = "mmm/research-evidence-document-v1"
_EVIDENCE_PAGE_SCHEMA = "mmm/research-evidence-page-v1"
# v3 invalidates old terminal-gap/synthesis caches created before failure-state and
# procedure preservation became part of the canonical contract.
_DOMAIN_CHECKPOINT_SCHEMA = "mmm/research-domain-checkpoint-v6"
_PAGE_PROTOCOL_SCHEMA = "mmm/research-page-host-completion-v4"
_SYNTHESIS_PROTOCOL_SCHEMA = "mmm/research-hierarchical-synthesis-v4"
_SYNTHESIS_INPUT_BYTES = 3_600
_SYNTHESIS_GROUP_ITEMS = 4
_MIN_ADAPTIVE_FRAGMENT_CHARS = 512
_MIN_CONTINUATION_PROGRESS_CHARS = 512
_CHECKPOINT_LOCK = threading.RLock()
_DOMAIN_LOCKS: dict[str, threading.Lock] = {}
_PROGRESS_LOCK = threading.RLock()
_PROGRESS_SEQUENCE = 0
_LATEST_PROGRESS: dict[str, Any] = {}
_PROGRESS_HOOK: Any | None = None


def set_research_progress_hook(hook: Any | None) -> None:
    """Install one optional observer; generation authority remains in this module."""

    global _PROGRESS_HOOK
    with _PROGRESS_LOCK:
        _PROGRESS_HOOK = hook if callable(hook) else None


def research_progress_snapshot() -> dict[str, Any]:
    with _PROGRESS_LOCK:
        return dict(_LATEST_PROGRESS)


def _emit_research_progress(event: str, **fields: Any) -> None:
    global _PROGRESS_SEQUENCE
    with _PROGRESS_LOCK:
        _PROGRESS_SEQUENCE += 1
        payload = {
            "sequence": _PROGRESS_SEQUENCE,
            "event": event,
            "monotonic_seconds": round(time.monotonic(), 3),
            **fields,
        }
        _LATEST_PROGRESS.clear()
        _LATEST_PROGRESS.update(payload)
        hook = _PROGRESS_HOOK
    print(
        "planner research progress: "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )
    if callable(hook):
        try:
            hook(dict(payload))
        except Exception:
            # Observability must never change research execution semantics.
            pass


def _emit_rag_trace(event: str, **fields: Any) -> None:
    print(
        "PRE-DESIGN RAG TRACE: "
        + json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def _research_page_messages(
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    page: Mapping[str, Any],
    continuation_offset: int = 0,
) -> list[dict[str, str]]:
    """Create one bounded page-reading request with a host-owned completion cursor."""

    system = (
        "You are reading exactly one bounded page from a host-owned Minecraft research "
        "evidence document. Extract only design-relevant claims supported by this page. "
        "Do not assume unseen pages are absent; the host will read every page and synthesize "
        "the page notes later. Return one compact JSON object matching research_note. "
        "research_note.domain_id must equal the assigned domain. Evidence refs should use "
        "the supplied page_ref. The host owns page delivery, cursor advancement, tail "
        "verification, and completion. Do not emit continuation, next_offset, tail_sha256, "
        "processed_span, or any other page-completion field. research_note.sufficient means "
        "this page produced at least one usable evidence-grounded claim; set it false when "
        "the page contains no such claim. It never controls pagination."
    )
    content = str(page.get("content", ""))
    offset = max(0, min(len(content), int(continuation_offset)))
    payload = {
        "authoritative_request": prompt,
        "domain": dict(domain),
        "evidence_document": _prompt_document_receipt(document),
        "evidence_page": {
            "schema_version": page.get("schema_version"),
            "page_ref": page.get("page_ref"),
            "unit_id": page.get("unit_id"),
            "part_index": page.get("part_index"),
            "part_count": page.get("part_count"),
            "content_start_offset": offset,
            "content_total_chars": len(content),
            "content_remaining": content[offset:],
        },
        "instruction": (
            "Read all of content_remaining in this call. Preserve source identifiers and "
            "concrete version/API facts, extract every supported design-relevant claim, and "
            "put unresolved page-local uncertainty in gaps. Return only research_note; the "
            "host records the raw page losslessly and completes this bounded page."
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


class _BoundedResearchOutputError(RuntimeError):
    pass


def _page_response_schema(research_note_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project only semantic research fields; mechanical page state is host-owned."""

    schema = json.loads(json.dumps(dict(research_note_schema)))
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("continuation", None)
    # Keep the loose research envelope parser-owned.  structured_output can then
    # canonicalize bare/aliased research notes locally without another model call.
    schema.pop("required", None)
    return schema


def _core_note(note: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve every model-authored semantic field used by downstream research/skills."""

    return {
        key: note[key]
        for key in (
            "domain_id",
            "claims",
            "gaps",
            "next_queries",
            "sufficient",
            "procedures",
        )
        if key in note
    }


def _validate_core_note(agentic_module: Any, note: Any, domain_id: str) -> dict[str, Any]:
    if not isinstance(note, Mapping):
        raise agentic_module.SpecValidationError("research_note must be an object.")
    raw = json.dumps({"research_note": _core_note(note)}, ensure_ascii=False)
    return agentic_module._parse_research_note(raw, domain_id)


def _parse_page_response(
    agentic_module: Any,
    raw: str,
    *,
    domain_id: str,
    current_offset: int,
    content_chars: int,
    tail_sha256: str,
) -> dict[str, Any]:
    note = agentic_module._parse_research_note(raw, domain_id)
    if current_offset < 0 or current_offset > content_chars:
        raise agentic_module.SpecValidationError(
            "host page cursor is outside the bounded evidence page."
        )
    # The complete content_remaining span was supplied in this request.  Page length and
    # tail hash are already exact host facts, so accepting a model-authored cursor can only
    # weaken the contract (and previously caused N -> N-1 -> N retry oscillation).
    return {
        "note": note,
        "continuation": {
            "complete": True,
            "next_offset": content_chars,
            "tail_sha256": tail_sha256,
        },
    }


def _validate_continuation_receipt(
    agentic_module: Any,
    continuation: Any,
    *,
    current_offset: int,
    content_chars: int,
    tail_sha256: str,
) -> dict[str, Any]:
    if not isinstance(continuation, Mapping):
        raise agentic_module.SpecValidationError("continuation must be an object.")
    allowed = {"complete", "next_offset", "tail_sha256"}
    if not allowed <= set(continuation) or (
        set(continuation) - allowed - {"legacy_adapter"}
    ):
        raise agentic_module.SpecValidationError(
            "continuation fields do not match the page protocol."
        )
    complete = continuation.get("complete")
    next_offset = continuation.get("next_offset")
    echoed_tail = str(continuation.get("tail_sha256", ""))
    if type(complete) is not bool or type(next_offset) is not int:
        raise agentic_module.SpecValidationError(
            "continuation complete/next_offset types are invalid."
        )
    if next_offset <= current_offset or next_offset > content_chars:
        raise agentic_module.SpecValidationError(
            "continuation did not make bounded forward progress."
        )
    minimum_progress = min(
        _MIN_CONTINUATION_PROGRESS_CHARS,
        max(0, content_chars - current_offset),
    )
    if next_offset - current_offset < minimum_progress:
        raise agentic_module.SpecValidationError(
            "continuation advanced less than the host-owned minimum page span."
        )
    if complete and (next_offset != content_chars or echoed_tail != tail_sha256):
        raise agentic_module.SpecValidationError(
            "continuation completed without the exact page tail receipt."
        )
    if not complete and next_offset >= content_chars:
        raise agentic_module.SpecValidationError(
            "continuation reached the tail but did not mark the page complete."
        )
    result = {
        "complete": complete,
        "next_offset": next_offset,
        "tail_sha256": echoed_tail,
    }
    if continuation.get("legacy_adapter") is True:
        result["legacy_adapter"] = True
    return result


def _repair_messages(
    messages: list[dict[str, str]],
    *,
    error: BaseException,
) -> list[dict[str, str]]:
    return [
        *messages,
        {
            "role": "system",
            "content": (
                "The prior response failed the bounded JSON contract: "
                f"{type(error).__name__}: {error}. Regenerate from the same supplied "
                "evidence span exactly once. Emit only schema-valid compact JSON. If you "
                "claim sufficient=true, include at least one concrete evidence-grounded "
                "claim. Do not repeat, quote, or continue the invalid output."
            ),
        },
    ]


def _emit_bounded_failure(
    event: str,
    *,
    progress_label: str,
    raw_output: str,
    error: BaseException,
) -> None:
    _emit_research_progress(
        event,
        label=progress_label,
        raw_output=raw_output,
        raw_output_sha256=_sha256_text(raw_output),
        raw_output_chars=len(raw_output),
        exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
        exception_message=str(error),
        traceback="".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    )


def _normalize_bounded_json_text(raw_text: str) -> str:
    """Recover one embedded or fenced JSON value locally without another model turn."""
    candidate = str(raw_text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    starts = [0] + [
        index for index, char in enumerate(candidate) if char in "{[" and index
    ]
    seen: set[int] = set()
    for position in starts:
        if position in seen:
            continue
        seen.add(position)
        try:
            value, _ = decoder.raw_decode(candidate[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (Mapping, list)):
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
    raise ValueError("no recoverable JSON object or array in bounded model output")


def _generate_bounded(
    agentic_module: Any,
    router: Any,
    *,
    messages: list[dict[str, str]],
    response_schema: Mapping[str, Any],
    parser: Any,
    progress_label: str,
) -> Any:
    """Use one model turn, deterministic host normalization, then fail closed."""
    _emit_research_progress("model_attempt", label=progress_label, attempt=1)
    raw = ""
    try:
        raw = router.generate_text(
            "planner",
            messages,
            response_format="json",
            response_schema=response_schema,
            tool_stage="research",
            enable_tools=False,
        )
    except Exception as model_error:
        _emit_bounded_failure(
            "bounded_model_failure",
            progress_label=progress_label,
            raw_output=raw,
            error=model_error,
        )
        raise
    try:
        return parser(raw)
    except Exception as first_error:
        _emit_bounded_failure(
            "bounded_parse_failure",
            progress_label=progress_label,
            raw_output=raw,
            error=first_error,
        )
        if not isinstance(first_error, agentic_module.SpecValidationError) and not _structured_output_failure(first_error):
            raise
    try:
        normalized = _normalize_bounded_json_text(raw)
        parsed = parser(normalized)
    except Exception as local_error:
        _emit_bounded_failure(
            "bounded_host_normalization_failure",
            progress_label=progress_label,
            raw_output=raw,
            error=local_error,
        )
        raise _BoundedResearchOutputError(
            "bounded JSON failed after one model attempt and deterministic host normalization: "
            f"{type(local_error).__name__}: {local_error}"
        ) from local_error
    _emit_research_progress(
        "bounded_host_normalized",
        label=progress_label,
        attempt=1,
        raw_output_sha256=_sha256_text(raw),
    )
    return parsed


def _structured_output_failure(exc: BaseException) -> bool:
    markers = (
        "jsondecodeerror",
        "unterminated string",
        "invalid json",
        "malformed json",
        "failed to parse json",
        "json response",
        "structured output",
    )
    pending: list[Any] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".casefold()
        if any(marker in text for marker in markers):
            return True
        pending.extend(
            (
                getattr(current, "cause", None),
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
            )
        )
    return False


def _router_role_signature(router: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "router": f"{type(router).__module__}.{type(router).__qualname__}",
        "profile": str(getattr(router, "profile", "")),
    }
    try:
        config = router.registry.role(router.profile, "planner")
    except Exception:
        return result
    for name in (
        "adapter",
        "provider",
        "model_id",
        "quantization",
        "max_context",
        "max_new_tokens",
    ):
        value = getattr(config, name, None)
        if value is not None:
            result[name] = value
    return result


def _domain_checkpoint_key(
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
) -> str:
    return _sha256(
        {
            "schema_version": _DOMAIN_CHECKPOINT_SCHEMA,
            "page_protocol": _PAGE_PROTOCOL_SCHEMA,
            "synthesis_protocol": _SYNTHESIS_PROTOCOL_SCHEMA,
            "role_signature": _router_role_signature(router),
            "prompt": prompt,
            "domain": dict(domain),
            "document_sha256": document.get("document_sha256"),
        }
    ).removeprefix("sha256:")


def _checkpoint_root() -> Path:
    configured = os.environ.get("MMM_RESEARCH_CHECKPOINT_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        output_root = os.environ.get("MMM_OUTPUT_ROOT", "").strip()
        workspace = os.environ.get("MMM_WORKSPACE", "").strip()
        if output_root:
            root = Path(output_root).expanduser() / "research-checkpoints-v3"
        elif workspace:
            workspace_path = Path(workspace).expanduser().resolve()
            if (workspace_path / ".git").exists():
                root = workspace_path.parent / "mmm-output" / "research-checkpoints-v3"
            else:
                root = workspace_path / "research-checkpoints-v3"
        else:
            root = Path(tempfile.gettempdir()) / "mmm-research-checkpoints-v3"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _checkpoint_dir(domain_key: str) -> Path:
    path = _checkpoint_root() / "domains-v3" / domain_key[:2] / domain_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _domain_lock(domain_key: str) -> threading.Lock:
    with _CHECKPOINT_LOCK:
        return _DOMAIN_LOCKS.setdefault(domain_key, threading.Lock())


def _unit_path(domain_key: str, kind: str, unit_key: str) -> Path:
    safe_key = unit_key.removeprefix("sha256:")
    return _checkpoint_dir(domain_key) / kind / f"{safe_key}.json"


def _read_unit(domain_key: str, kind: str, unit_key: str) -> Any | None:
    path = _unit_path(domain_key, kind, unit_key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("value")
    if (
        payload.get("schema_version") != _DOMAIN_CHECKPOINT_SCHEMA
        or payload.get("domain_key") != domain_key
        or payload.get("kind") != kind
        or payload.get("unit_key") != unit_key
        or payload.get("value_sha256") != _sha256(value)
    ):
        return None
    return value


def _write_unit(
    domain_key: str,
    kind: str,
    unit_key: str,
    value: Any,
) -> None:
    path = _unit_path(domain_key, kind, unit_key)
    payload = {
        "schema_version": _DOMAIN_CHECKPOINT_SCHEMA,
        "domain_key": domain_key,
        "kind": kind,
        "unit_key": unit_key,
        "value_sha256": _sha256(value),
        "value": value,
    }
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _manifest_path(domain_key: str) -> Path:
    return _checkpoint_dir(domain_key) / "manifest.json"


def _read_complete_manifest(
    agentic_module: Any,
    domain_key: str,
    domain_id: str,
) -> dict[str, Any] | None:
    """Reuse successful work only. Failed/terminal research is always recomputed."""

    path = _manifest_path(domain_key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("status") != "complete":
        return None
    note = payload.get("note")
    if (
        payload.get("schema_version") != _DOMAIN_CHECKPOINT_SCHEMA
        or payload.get("domain_key") != domain_key
        or payload.get("note_sha256") != _sha256(note)
    ):
        return None
    try:
        validated = _validate_core_note(agentic_module, note, domain_id)
        if validated.get("sufficient") is not True or not validated.get("claims"):
            return None
        _validate_note_artifacts(domain_key, note)
    except Exception:
        return None
    return dict(note) if isinstance(note, Mapping) else None


def _validate_note_artifacts(domain_key: str, note: Any) -> None:
    if not isinstance(note, Mapping):
        raise ValueError("checkpoint note is not an object")
    directory = _checkpoint_dir(domain_key).resolve()
    contracts = (
        ("claim_catalog", "catalog_sha256", "claim_count"),
        ("evidence_ledger", "ledger_sha256", "record_count"),
    )
    for field, hash_field, count_field in contracts:
        receipt = note.get(field)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"checkpoint {field} receipt is missing")
        path = Path(str(receipt.get("path", ""))).expanduser().resolve()
        if path.parent != directory or not path.is_file():
            raise ValueError(f"checkpoint {field} path escaped its domain directory")
        content = path.read_text(encoding="utf-8")
        if receipt.get(hash_field) != _sha256_text(content):
            raise ValueError(f"checkpoint {field} hash mismatch")
        records = []
        for line in content.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"checkpoint {field} record is invalid")
            records.append(value)
        if int(receipt.get(count_field, -1)) != len(records):
            raise ValueError(f"checkpoint {field} count mismatch")


def _write_manifest(
    domain_key: str,
    *,
    status: str,
    note: Mapping[str, Any],
    failures: list[dict[str, str]],
) -> None:
    payload = {
        "schema_version": _DOMAIN_CHECKPOINT_SCHEMA,
        "domain_key": domain_key,
        "status": status,
        "note_sha256": _sha256(note),
        "note": dict(note),
        "failures": failures,
    }
    _atomic_write_text(
        _manifest_path(domain_key),
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _page_claims_with_provenance(
    note: Mapping[str, Any],
    *,
    page_ref: str,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for claim in note.get("claims", []):
        if not isinstance(claim, Mapping):
            continue
        refs = [
            str(item).strip()
            for item in claim.get("evidence_refs", [])
            if str(item).strip()
        ]
        if page_ref and page_ref not in refs:
            refs.insert(0, page_ref)
        text = str(claim.get("claim", "")).strip()
        if text:
            claims.append({"claim": text, "evidence_refs": refs})
    return claims


def _fragment_page(
    page: Mapping[str, Any],
    *,
    content: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    value = dict(page)
    parent_ref = str(page.get("page_ref", ""))
    value.update(
        {
            "content": content,
            "page_ref": f"{parent_ref}#fragment={start}:{end}",
            "fragment_start": start,
            "fragment_end": end,
        }
    )
    return value


def _read_page_losslessly(
    agentic_module: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    page: Mapping[str, Any],
    domain_key: str,
    progress_label: str,
    failures: list[dict[str, str]],
) -> list[dict[str, Any]]:
    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    content = str(page.get("content", ""))
    page_ref = str(page.get("page_ref", "")).strip()
    tail_sha256 = _sha256_text(content)
    offset = 0
    notes: list[dict[str, Any]] = []

    while offset < len(content) or (not content and offset == 0):
        unit_key = _sha256(
            {
                "protocol": _PAGE_PROTOCOL_SCHEMA,
                "page_ref": page_ref,
                "content_sha256": tail_sha256,
                "offset": offset,
            }
        )
        value = _read_unit(domain_key, "page", unit_key)
        parsed: dict[str, Any] | None = None
        if isinstance(value, Mapping):
            try:
                cached_note = _validate_core_note(
                    agentic_module, value.get("note"), domain_id
                )
                cached_continuation = _validate_continuation_receipt(
                    agentic_module,
                    value.get("continuation"),
                    current_offset=offset,
                    content_chars=len(content),
                    tail_sha256=tail_sha256,
                )
                parsed = {
                    "note": cached_note,
                    "continuation": cached_continuation,
                }
            except Exception:
                parsed = None
        if parsed is not None:
            _emit_research_progress(
                "page_checkpoint_hit",
                label=progress_label,
                offset=offset,
            )
        else:
            messages = _research_page_messages(
                prompt=prompt,
                domain=domain,
                document=document,
                page=page,
                continuation_offset=offset,
            )
            try:
                parsed = _generate_bounded(
                    agentic_module,
                    router,
                    messages=messages,
                    response_schema=_page_response_schema(
                        agentic_module._RESEARCH_NOTE_SCHEMA
                    ),
                    parser=lambda raw: _parse_page_response(
                        agentic_module,
                        raw,
                        domain_id=domain_id,
                        current_offset=offset,
                        content_chars=len(content),
                        tail_sha256=tail_sha256,
                    ),
                    progress_label=f"{progress_label} offset {offset}",
                )
            except _BoundedResearchOutputError as exc:
                remaining = content[offset:]
                if len(remaining) > _MIN_ADAPTIVE_FRAGMENT_CHARS:
                    midpoint = offset + max(1, len(remaining) // 2)
                    left = _fragment_page(
                        page,
                        content=content[offset:midpoint],
                        start=offset,
                        end=midpoint,
                    )
                    right = _fragment_page(
                        page,
                        content=content[midpoint:],
                        start=midpoint,
                        end=len(content),
                    )
                    _emit_research_progress(
                        "page_adaptive_split",
                        label=progress_label,
                        start_offset=offset,
                        midpoint=midpoint,
                        end_offset=len(content),
                    )
                    return notes + _read_page_losslessly(
                        agentic_module,
                        router,
                        prompt=prompt,
                        domain=domain,
                        document=document,
                        page=left,
                        domain_key=domain_key,
                        progress_label=f"{progress_label}.left",
                        failures=failures,
                    ) + _read_page_losslessly(
                        agentic_module,
                        router,
                        prompt=prompt,
                        domain=domain,
                        document=document,
                        page=right,
                        domain_key=domain_key,
                        progress_label=f"{progress_label}.right",
                        failures=failures,
                    )
                failures.append(
                    {
                        "unit": page_ref or progress_label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return notes
            _write_unit(domain_key, "page", unit_key, parsed)

        note = dict(parsed["note"])
        notes.append(
            {
                **note,
                "page_ref": page_ref,
                "unit_id": str(page.get("unit_id", "")),
                "claims": _page_claims_with_provenance(note, page_ref=page_ref),
            }
        )
        continuation = parsed["continuation"]
        next_offset = int(continuation.get("next_offset", len(content)))
        if bool(continuation.get("complete")):
            return notes
        if next_offset <= offset:
            failures.append(
                {
                    "unit": page_ref or progress_label,
                    "error": "cached continuation made no forward progress",
                }
            )
            return notes
        offset = next_offset
    return notes


def _group_synthesis_notes(notes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 2
    for note in notes:
        size = len(
            json.dumps(note, ensure_ascii=False, sort_keys=True, default=str).encode(
                "utf-8"
            )
        )
        if current and (
            len(current) >= _SYNTHESIS_GROUP_ITEMS
            or current_bytes + size > _SYNTHESIS_INPUT_BYTES
        ):
            groups.append(current)
            current = []
            current_bytes = 2
        current.append(note)
        current_bytes += size + 1
    if current:
        groups.append(current)

    if len(notes) > 1 and len(groups) == len(notes):
        atomic: list[dict[str, Any]] = []
        for note in notes:
            domain_id = str(note.get("domain_id", "unknown"))
            sufficient = bool(note.get("sufficient"))
            for claim in note.get("claims", []):
                atomic.append(
                    {
                        "domain_id": domain_id,
                        "claims": [claim],
                        "gaps": [],
                        "next_queries": [],
                        "procedures": [],
                        "sufficient": sufficient,
                    }
                )
            for gap in note.get("gaps", []):
                atomic.append(
                    {
                        "domain_id": domain_id,
                        "claims": [],
                        "gaps": [gap],
                        "next_queries": [],
                        "procedures": [],
                        "sufficient": sufficient,
                    }
                )
            for query in note.get("next_queries", []):
                atomic.append(
                    {
                        "domain_id": domain_id,
                        "claims": [],
                        "gaps": [],
                        "next_queries": [query],
                        "procedures": [],
                        "sufficient": sufficient,
                    }
                )
            for procedure in note.get("procedures", []):
                atomic.append(
                    {
                        "domain_id": domain_id,
                        "claims": [],
                        "gaps": [],
                        "next_queries": [],
                        "procedures": [procedure],
                        "sufficient": sufficient,
                    }
                )
        if atomic:
            return [
                atomic[index : index + _SYNTHESIS_GROUP_ITEMS]
                for index in range(0, len(atomic), _SYNTHESIS_GROUP_ITEMS)
            ]
        return groups
    return groups


def _synthesis_messages(
    *,
    prompt: str,
    domain: Mapping[str, Any],
    notes: list[dict[str, Any]],
    level: int,
    group_index: int,
) -> list[dict[str, str]]:
    payload = {
        "authoritative_request_receipt": {
            "sha256": _sha256_text(prompt),
            "char_count": len(prompt),
        },
        "domain": _bounded_domain_projection(domain),
        "protocol": _SYNTHESIS_PROTOCOL_SCHEMA,
        "level": level,
        "group_index": group_index,
        "bounded_child_notes": notes,
        "instruction": (
            "Synthesize only these child notes into one compact research_note. Preserve "
            "concrete evidence-grounded claims, contradictions, gaps, next queries, and any "
            "cited reusable procedures with their requires/provides edges. The full lossless "
            "evidence ledger is retained by the host; this response is a bounded design "
            "summary, not a replacement. Set sufficient=true only when at least one concrete "
            "grounded claim survives synthesis and the supplied evidence resolves this group."
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a bounded hierarchical Minecraft research synthesizer. Return "
                "only one JSON object matching research_note. Do not use tools and do not "
                "repeat raw evidence. Preserve cited procedures instead of silently dropping "
                "them."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _bounded_text_receipt(value: str, *, max_bytes: int = 384) -> Any:
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    prefix = _split_utf8_text(value, max_bytes)[0]
    return {
        "externalized_to_request_ledger": True,
        "text_sha256": _sha256_text(value),
        "text_chars": len(value),
        "text_bytes": len(value.encode("utf-8")),
        "prefix": prefix,
    }


def _bounded_domain_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_domain_projection(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if len(rendered.encode("utf-8")) > 1_024:
            return {
                "externalized_to_request_ledger": True,
                "value_sha256": _sha256(value),
                "item_count": len(value),
                "sample": [_bounded_domain_projection(item) for item in value[:2]],
            }
        return [_bounded_domain_projection(item) for item in value]
    if isinstance(value, tuple):
        return [_bounded_domain_projection(item) for item in value]
    if isinstance(value, str):
        return _bounded_text_receipt(value)
    return value


def _split_synthesis_leaf(note: Mapping[str, Any]) -> list[dict[str, Any]]:
    fragment = note.get("evidence_fragment")
    if not isinstance(fragment, Mapping):
        return []
    content = str(fragment.get("content", ""))
    if len(content) <= _MIN_ADAPTIVE_FRAGMENT_CHARS:
        return []
    midpoint = len(content) // 2
    result: list[dict[str, Any]] = []
    for label, start, end in (
        ("left", 0, midpoint),
        ("right", midpoint, len(content)),
    ):
        child_content = content[start:end]
        child_fragment = {
            **dict(fragment),
            "page_ref": f"{fragment.get('page_ref', '')}#synthesis-{label}={start}:{end}",
            "content_sha256": _sha256_text(child_content),
            "content": child_content,
        }
        result.append({**dict(note), "evidence_fragment": child_fragment})
    return result


def _synthesize_group_with_recovery(
    agentic_module: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    group: list[dict[str, Any]],
    domain_key: str,
    failures: list[dict[str, str]],
    level: int,
    group_label: str,
) -> list[dict[str, Any]]:
    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    unit_key = _sha256(
        {
            "protocol": _SYNTHESIS_PROTOCOL_SCHEMA,
            "level": level,
            "group": group,
        }
    )
    cached = _read_unit(domain_key, "synthesis", unit_key)
    note: dict[str, Any] | None = None
    if isinstance(cached, Mapping):
        try:
            note = _validate_core_note(agentic_module, cached, domain_id)
            if note.get("sufficient") is True and not note.get("claims"):
                note = None
        except Exception:
            note = None
    if note is not None:
        _emit_research_progress(
            "synthesis_checkpoint_hit",
            domain_id=domain_id,
            level=level,
            group=group_label,
            checkpoint_unit=str(_unit_path(domain_key, "synthesis", unit_key)),
        )
        return [note]
    try:
        note = _generate_bounded(
            agentic_module,
            router,
            messages=_synthesis_messages(
                prompt=prompt,
                domain=domain,
                notes=group,
                level=level,
                group_index=0,
            ),
            response_schema=agentic_module._RESEARCH_NOTE_SCHEMA,
            parser=lambda raw: agentic_module._parse_research_note(raw, domain_id),
            progress_label=f"domain {domain_id} synthesis {level}:{group_label}",
        )
        if note.get("sufficient") is True and not note.get("claims"):
            raise agentic_module.SpecValidationError(
                "research synthesis declared sufficient=true without a grounded claim"
            )
        _write_unit(domain_key, "synthesis", unit_key, note)
        return [note]
    except _BoundedResearchOutputError as exc:
        children: list[list[dict[str, Any]]] = []
        if len(group) > 1:
            midpoint = len(group) // 2
            children = [group[:midpoint], group[midpoint:]]
        elif group:
            split_leaf = _split_synthesis_leaf(group[0])
            children = [[item] for item in split_leaf]
        if children:
            _emit_research_progress(
                "synthesis_adaptive_split",
                domain_id=domain_id,
                level=level,
                group=group_label,
                child_count=len(children),
                reason=f"{type(exc).__name__}: {exc}",
            )
            recovered: list[dict[str, Any]] = []
            for child_index, child in enumerate(children):
                recovered.extend(
                    _synthesize_group_with_recovery(
                        agentic_module,
                        router,
                        prompt=prompt,
                        domain=domain,
                        group=child,
                        domain_key=domain_key,
                        failures=failures,
                        level=level,
                        group_label=f"{group_label}.{child_index}",
                    )
                )
            return recovered
        failure = {
            "unit": f"synthesis:{level}:{group_label}",
            "error": f"{type(exc).__name__}: {exc}",
        }
        failures.append(failure)
        _emit_research_progress(
            "synthesis_terminal_failure",
            domain_id=domain_id,
            level=level,
            group=group_label,
            failure=failure,
            checkpoint_dir=str(_checkpoint_dir(domain_key)),
        )
        return [
            {
                "domain_id": domain_id,
                "claims": [],
                "gaps": ["A bounded synthesis page failed validation."],
                "next_queries": [],
                "procedures": [],
                "sufficient": False,
            }
        ]


def _hierarchical_synthesis(
    agentic_module: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    page_notes: list[dict[str, Any]],
    domain_key: str,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    current = page_notes or [
        {
            "domain_id": domain_id,
            "claims": [],
            "gaps": ["No readable evidence page note was produced."],
            "next_queries": [],
            "procedures": [],
            "sufficient": False,
        }
    ]
    level = 0
    while True:
        groups = _group_synthesis_notes(current)
        next_level: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups):
            next_level.extend(
                _synthesize_group_with_recovery(
                    agentic_module,
                    router,
                    prompt=prompt,
                    domain=domain,
                    group=group,
                    domain_key=domain_key,
                    failures=failures,
                    level=level,
                    group_label=str(group_index),
                )
            )
        if len(next_level) == 1:
            return next_level[0]
        if failures and len(next_level) >= len(current):
            return {
                "domain_id": domain_id,
                "claims": [],
                "gaps": [
                    "Bounded research synthesis reached a validated no-progress frontier; "
                    "full evidence remains in the durable ledger."
                ],
                "next_queries": [],
                "procedures": [],
                "sufficient": False,
            }
        current = next_level
        level += 1


def _stable_unique_claims(page_notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for note in page_notes:
        for claim in note.get("claims", []):
            if not isinstance(claim, Mapping):
                continue
            value = {
                "claim": str(claim.get("claim", "")).strip(),
                "evidence_refs": [
                    str(item).strip()
                    for item in claim.get("evidence_refs", [])
                    if str(item).strip()
                ],
            }
            key = _sha256(value)
            if not value["claim"] or key in seen:
                continue
            seen.add(key)
            result.append(value)
    return result


def _host_page_note(domain_id: str, page: Mapping[str, Any]) -> dict[str, Any]:
    content = str(page.get("content", ""))
    page_ref = str(page.get("page_ref", ""))
    return {
        "domain_id": domain_id,
        "claims": [],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
        "evidence_fragment": {
            "page_ref": page_ref,
            "unit_id": str(page.get("unit_id", "")),
            "part_index": page.get("part_index"),
            "part_count": page.get("part_count"),
            "content_sha256": _sha256_text(content),
            "content_chars": len(content),
            "content": content,
        },
    }


def _materialize_claim_catalog(
    domain_key: str,
    domain_id: str,
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    path = _checkpoint_dir(domain_key) / "claims.jsonl"
    content = "".join(
        json.dumps(claim, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for claim in claims
    )
    _atomic_write_text(path, content)
    return {
        "schema_version": "mmm/research-claim-catalog-v1",
        "domain_id": domain_id,
        "claim_count": len(claims),
        "catalog_sha256": _sha256_text(content),
        "path": str(path),
    }


def _materialize_evidence_ledger(
    domain_key: str,
    domain_id: str,
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    path = _checkpoint_dir(domain_key) / "evidence-ledger.jsonl"
    records = [
        {
            "page_ref": str(page.get("page_ref", "")),
            "unit_id": str(page.get("unit_id", "")),
            "part_index": page.get("part_index"),
            "part_count": page.get("part_count"),
            "content_sha256": _sha256_text(str(page.get("content", ""))),
            "content": str(page.get("content", "")),
        }
        for page in pages
    ]
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    _atomic_write_text(path, content)
    return {
        "schema_version": "mmm/research-evidence-ledger-v1",
        "domain_id": domain_id,
        "record_count": len(records),
        "ledger_sha256": _sha256_text(content),
        "path": str(path),
    }


def _research_document_domain(
    agentic_module: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del trace_metadata
    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    domain_key = _domain_checkpoint_key(
        router,
        prompt=prompt,
        domain=domain,
        document=document,
    )
    with _domain_lock(domain_key):
        cached = _read_complete_manifest(agentic_module, domain_key, domain_id)
        if cached is not None:
            _emit_research_progress(
                "domain_checkpoint_complete",
                domain_id=domain_id,
                manifest_path=str(_manifest_path(domain_key)),
                checkpoint_dir=str(_checkpoint_dir(domain_key)),
                note=cached,
            )
            return cached

        pages = _read_evidence_pages(document)
        failures: list[dict[str, str]] = []
        page_notes: list[dict[str, Any]] = []
        _emit_research_progress(
            "domain_start",
            domain_id=domain_id,
            page_count=len(pages),
            evidence_document=_prompt_document_receipt(document),
            evidence_pages_path=document.get("pages_path"),
            evidence_raw_path=document.get("raw_path"),
            checkpoint_dir=str(_checkpoint_dir(domain_key)),
        )
        for page_index, page in enumerate(pages):
            _emit_research_progress(
                "page_start",
                domain_id=domain_id,
                page_index=page_index + 1,
                page_count=len(pages),
                page_ref=str(page.get("page_ref", "")),
            )
            page_notes.append(_host_page_note(domain_id, page))
            _emit_research_progress(
                "page_ledgered",
                domain_id=domain_id,
                page_index=page_index + 1,
                page_count=len(pages),
                page_ref=str(page.get("page_ref", "")),
            )

        summary = _hierarchical_synthesis(
            agentic_module,
            router,
            prompt=prompt,
            domain=domain,
            page_notes=page_notes,
            domain_key=domain_key,
            failures=failures,
        )
        claims = _stable_unique_claims([*page_notes, summary])
        catalog = _materialize_claim_catalog(domain_key, domain_id, claims)
        evidence_ledger = _materialize_evidence_ledger(domain_key, domain_id, pages)
        failure_reasons = []
        if failures:
            failure_reasons.append("bounded synthesis failure")
        if summary.get("sufficient") is not True:
            failure_reasons.append("synthesis returned sufficient=false")
        if not claims:
            failure_reasons.append("synthesis produced zero grounded claims")

        status = "failed" if failure_reasons else "complete"
        note: dict[str, Any] = {
            **_core_note(summary),
            "evidence_document": _prompt_document_receipt(document),
            "claim_catalog": catalog,
            "evidence_ledger": evidence_ledger,
            "checkpoint": {
                "schema_version": _DOMAIN_CHECKPOINT_SCHEMA,
                "request_sha256": "sha256:" + domain_key,
                "status": status,
                "manifest_path": str(_manifest_path(domain_key)),
                "checkpoint_dir": str(_checkpoint_dir(domain_key)),
            },
        }
        if failures:
            existing_gaps = [str(item) for item in note.get("gaps", [])]
            note["gaps"] = existing_gaps + [
                f"{item['unit']}: {item['error']}" for item in failures
            ]
            note["research_failures"] = list(failures)
        if failure_reasons:
            note["sufficient"] = False
            note["fixed_point"] = True
            note["failure_reasons"] = failure_reasons

        _write_manifest(
            domain_key,
            status=status,
            note=note,
            failures=failures,
        )

        if status != "complete":
            _emit_research_progress(
                "domain_failure",
                domain_id=domain_id,
                status=status,
                failure_reasons=failure_reasons,
                failures=failures,
                summary=summary,
                claim_catalog=catalog,
                evidence_ledger=evidence_ledger,
                evidence_document=_prompt_document_receipt(document),
                manifest_path=str(_manifest_path(domain_key)),
                checkpoint_dir=str(_checkpoint_dir(domain_key)),
                note=note,
            )
            raise _BoundedResearchOutputError(
                "pre-design research failed closed for domain "
                f"{domain_id!r}: {'; '.join(failure_reasons)}; "
                f"manifest={_manifest_path(domain_key)}"
            )

        _emit_research_progress(
            "domain_complete",
            domain_id=domain_id,
            status=status,
            claim_count=catalog["claim_count"],
            procedure_count=len(note.get("procedures", [])),
            page_count=len(pages),
            failure_count=0,
            claim_catalog=catalog,
            evidence_ledger=evidence_ledger,
            manifest_path=str(_manifest_path(domain_key)),
            checkpoint_dir=str(_checkpoint_dir(domain_key)),
        )
        return note


def _materialize_domain_evidence_document(
    domain_id: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    raw_text = json.dumps(
        dict(evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    document_sha256 = _sha256_text(raw_text)
    digest = document_sha256.removeprefix("sha256:")
    safe_domain = _safe_name(domain_id)
    directory = _evidence_root() / digest
    raw_path = directory / f"{safe_domain}.json"
    pages_path = directory / f"{safe_domain}.pages.jsonl"

    units = list(_evidence_units(evidence))
    rendered_units = [
        json.dumps(
            {"unit_id": unit_id, "value": value},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for unit_id, value in units
    ]
    pages: list[dict[str, Any]] = []
    packed: list[str] = []
    packed_bytes = 2

    def flush_packed() -> None:
        nonlocal packed, packed_bytes
        if not packed:
            return
        content = "[" + ",".join(packed) + "]"
        pages.append(
            {
                "schema_version": _EVIDENCE_PAGE_SCHEMA,
                "domain_id": domain_id,
                "unit_id": f"packed:{len(pages)}",
                "part_index": 0,
                "part_count": 1,
                "content": content,
            }
        )
        packed = []
        packed_bytes = 2

    for rendered in rendered_units:
        rendered_bytes = len(rendered.encode("utf-8"))
        if rendered_bytes + 2 > _EVIDENCE_PAGE_CHARS:
            flush_packed()
            parts = _split_utf8_text(rendered, _EVIDENCE_PAGE_CHARS)
            record_sha256 = _sha256_text(rendered)
            char_offset = 0
            for part_index, content in enumerate(parts):
                next_offset = char_offset + len(content)
                pages.append(
                    {
                        "schema_version": _EVIDENCE_PAGE_SCHEMA,
                        "domain_id": domain_id,
                        "unit_id": f"oversize:{len(pages)}",
                        "part_index": part_index,
                        "part_count": len(parts),
                        "record_sha256": record_sha256,
                        "char_start": char_offset,
                        "char_end": next_offset,
                        "content": content,
                    }
                )
                char_offset = next_offset
            continue
        added = rendered_bytes + (1 if packed else 0)
        if packed and packed_bytes + added > _EVIDENCE_PAGE_CHARS:
            flush_packed()
        packed.append(rendered)
        packed_bytes += rendered_bytes + (1 if len(packed) > 1 else 0)
    flush_packed()

    if not pages:
        pages.append(
            {
                "schema_version": _EVIDENCE_PAGE_SCHEMA,
                "domain_id": domain_id,
                "unit_id": "empty",
                "part_index": 0,
                "part_count": 1,
                "content": "{}",
            }
        )

    page_count = len(pages)
    for page_index, page in enumerate(pages):
        page["page_index"] = page_index
        page["page_count"] = page_count
        page["page_ref"] = f"{document_sha256}#page={page_index + 1}/{page_count}"

    pages_text = "\n".join(
        json.dumps(page, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for page in pages
    ) + "\n"
    _atomic_write_text(raw_path, raw_text)
    _atomic_write_text(pages_path, pages_text)
    _emit_rag_trace(
        "evidence_document_materialized",
        domain_id=domain_id,
        document_sha256=document_sha256,
        raw_source_keys=sorted(str(key) for key in evidence),
        model_unit_count=len(units),
        page_count=page_count,
        pages=[
            {
                "page_ref": str(page.get("page_ref") or ""),
                "unit_id": str(page.get("unit_id") or ""),
                "content_chars": len(str(page.get("content") or "")),
                "content_sha256": _sha256_text(str(page.get("content") or "")),
            }
            for page in pages
        ],
    )

    return {
        "schema_version": _EVIDENCE_DOCUMENT_SCHEMA,
        "domain_id": domain_id,
        "document_sha256": document_sha256,
        "raw_path": str(raw_path),
        "pages_path": str(pages_path),
        "page_count": page_count,
        "page_chars": _EVIDENCE_PAGE_CHARS,
        "page_bytes": _EVIDENCE_PAGE_CHARS,
        "source_keys": sorted(str(key) for key in evidence),
        "model_projection": "claim_bearing_source_bodies_only;raw_receipt_lossless",
    }


def _split_utf8_text(value: str, max_bytes: int) -> list[str]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if not value:
        return [""]
    parts: list[str] = []
    start = 0
    while start < len(value):
        low = start + 1
        high = len(value)
        best = start
        while low <= high:
            midpoint = (low + high) // 2
            if len(value[start:midpoint].encode("utf-8")) <= max_bytes:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best == start:
            raise ValueError("one Unicode scalar exceeds the evidence byte budget")
        parts.append(value[start:best])
        start = best
    return parts


def _evidence_units(evidence: Mapping[str, Any]):
    """Project claim-bearing source bodies to the model; keep diagnostics host-only.

    The raw evidence receipt remains losslessly persisted by
    ``_materialize_domain_evidence_document``.  The model projection must never serialize
    query/provider/fusion envelopes as evidence pages because those diagnostics can bury a
    real source body across many 1.8KB fragments and make the model reason over retrieval
    traces instead of source text.
    """

    for source_key, value in evidence.items():
        if str(source_key) == "grounded_rag" and isinstance(value, Mapping):
            queries = value.get("queries")
            if isinstance(queries, list):
                admitted = 0
                for query_index, raw_query in enumerate(queries):
                    if not isinstance(raw_query, Mapping):
                        _emit_rag_trace(
                            "evidence_query_skipped",
                            source=str(source_key),
                            query_index=query_index,
                            reason="query_not_mapping",
                        )
                        continue
                    query = str(raw_query.get("query") or "").strip()
                    records = raw_query.get("evidence_records")
                    if not isinstance(records, list) or not records:
                        _emit_rag_trace(
                            "evidence_query_no_source_body_records",
                            source=str(source_key),
                            query_index=query_index,
                            query=query,
                            provider_status=str(raw_query.get("github_provider_status") or ""),
                            saturation_reason=str(raw_query.get("github_saturation_reason") or ""),
                        )
                        continue
                    for record_index, raw_record in enumerate(records):
                        if not isinstance(raw_record, Mapping):
                            _emit_rag_trace(
                                "evidence_record_skipped",
                                source=str(source_key),
                                query=query,
                                query_index=query_index,
                                record_index=record_index,
                                reason="record_not_mapping",
                            )
                            continue
                        body = str(
                            raw_record.get("content")
                            or raw_record.get("body")
                            or raw_record.get("text")
                            or ""
                        ).strip()
                        source_id = str(raw_record.get("source_id") or "").strip()
                        if not body:
                            _emit_rag_trace(
                                "evidence_record_skipped",
                                source=str(source_key),
                                query=query,
                                query_index=query_index,
                                record_index=record_index,
                                source_id=source_id,
                                reason="no_source_body",
                            )
                            continue
                        digest = str(raw_record.get("content_sha256") or "").strip() or _sha256_text(body)
                        metadata = raw_record.get("metadata")
                        safe_metadata: dict[str, Any] = {}
                        if isinstance(metadata, Mapping):
                            for key in ("repository", "default_branch", "readme_path", "path", "file_path"):
                                item = metadata.get(key)
                                if item not in (None, ""):
                                    safe_metadata[key] = item
                        projected = {
                            "query": query,
                            "source_id": source_id,
                            "source_type": str(raw_record.get("source_type") or ""),
                            "source_locator": str(raw_record.get("source_locator") or ""),
                            "url": str(raw_record.get("url") or ""),
                            "title": str(raw_record.get("title") or ""),
                            "content_sha256": digest,
                            "content": body,
                            "retrieval_section": str(raw_record.get("retrieval_section") or ""),
                            "evidence_origin": str(raw_record.get("evidence_origin") or ""),
                            "metadata": safe_metadata,
                        }
                        admitted += 1
                        _emit_rag_trace(
                            "evidence_record_admitted",
                            source=str(source_key),
                            query=query,
                            query_index=query_index,
                            record_index=record_index,
                            source_id=source_id,
                            body_chars=len(body),
                            body_sha256=digest,
                        )
                        yield f"{source_key}:source:{query_index}:{record_index}", projected
                _emit_rag_trace(
                    "grounded_rag_model_projection_complete",
                    source=str(source_key),
                    query_count=len(queries),
                    source_body_unit_count=admitted,
                    diagnostic_envelopes_excluded=True,
                )
                continue
        yield str(source_key), value


def _read_evidence_pages(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(document.get("pages_path", ""))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Research evidence pages are missing: {path}")
    pages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid research evidence page in {path}")
            content = str(value.get("content", ""))
            if len(content.encode("utf-8")) > _EVIDENCE_PAGE_CHARS:
                raise ValueError(
                    f"Research evidence page exceeds {_EVIDENCE_PAGE_CHARS} bytes: {path}"
                )
            pages.append(value)
    expected = int(document.get("page_count", -1))
    if expected != len(pages):
        raise ValueError(
            f"Research evidence page count mismatch: expected {expected}, got {len(pages)}"
        )
    return pages


def _prompt_document_receipt(document: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "domain_id",
        "document_sha256",
        "page_count",
        "page_chars",
        "source_keys",
    )
    return {key: document[key] for key in keep if key in document}


def _evidence_root() -> Path:
    configured = os.environ.get("MMM_RESEARCH_DOCUMENT_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        workspace = os.environ.get("MMM_WORKSPACE", "").strip()
        base = Path(workspace).expanduser() if workspace else Path.cwd()
        root = base / "mmm-output" / "research-evidence"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _safe_name(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value.strip()
    )
    return cleaned[:96] or "unknown"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == content:
                return
        except OSError:
            pass
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temp_name = handle.name
        os.replace(temp_name, path)
    finally:
        if temp_name:
            temp_path = Path(temp_name)
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)


def _forced_rag_bundle(router: Any, research_brief: Mapping[str, Any]) -> dict[str, Any]:
    raw_domains = research_brief.get("domains")
    domains = [item for item in raw_domains or [] if isinstance(item, Mapping)]
    jobs: list[tuple[str, str]] = []
    for domain in domains:
        domain_id = str(domain.get("domain_id", "")).strip()
        queries = domain.get("queries")
        if not domain_id or not isinstance(queries, list):
            continue
        for query in queries:
            query_text = str(query).strip()
            if query_text:
                jobs.append((domain_id, query_text))

    versions = _research_versions(router)
    code_index = _existing_code_index()
    worker_count = max(1, min(8, len(jobs)))

    def run(job: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        domain_id, query = job
        project_results = _search_authoritative_catalog(query, versions)
        code_result = _search_code_index(code_index, query)
        return domain_id, {
            "query": query,
            "query_sha256": _sha256_text(query),
            "project_rag": project_results,
            "code_rag": code_result,
        }

    by_domain: dict[str, list[dict[str, Any]]] = {
        str(item.get("domain_id", "")): [] for item in domains
    }
    if jobs:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="mmm_pre_design_rag",
        ) as pool:
            for domain_id, result in pool.map(run, jobs):
                by_domain.setdefault(domain_id, []).append(result)

    payload = {
        "schema_version": "mmm/forced-pre-design-rag-v2",
        "versions": list(versions),
        "domain_count": len(domains),
        "query_count": len(jobs),
        "project_source_count": sum(
            len(item.get("project_rag", {}).get("sources", []))
            for values in by_domain.values()
            for item in values
        ),
        "code_index_status": "available" if code_index is not None else "not_indexed",
        "code_index_path": str(code_index) if code_index is not None else "",
        "domains": [
            {
                "domain_id": str(domain.get("domain_id", "")),
                "queries": by_domain.get(str(domain.get("domain_id", "")), []),
            }
            for domain in domains
        ],
    }
    payload["research_sha256"] = _sha256(payload)
    return payload


def _research_versions(router: Any) -> tuple[str, ...]:
    requested = str(
        getattr(router, "_mmm_requested_minecraft_version", "") or ""
    ).strip()
    existing = str(
        getattr(router, "_mmm_existing_minecraft_version", "") or ""
    ).strip()
    if requested:
        return (requested,)
    if existing:
        return (existing,)
    return ()


def _search_authoritative_catalog(
    query: str,
    versions: tuple[str, ...],
) -> dict[str, Any]:
    retriever = AuthoritativeEvidenceRetriever()
    sources: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    if not versions:
        try:
            catalog = target_neutral_evidence_catalog()
            limit = min(6, len(catalog))
            for source in retriever.search(query, limit=limit):
                payload = asdict(source)
                payload["matched_version"] = ""
                sources.setdefault(source.source_id, payload)
        except Exception as exc:
            errors.append(
                {
                    "minecraft_version": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    for version in versions:
        try:
            catalog = evidence_catalog_for_version(version)
            limit = min(6, len(catalog))
            for source in retriever.search(
                query,
                minecraft_version=version,
                limit=limit,
            ):
                payload = asdict(source)
                payload["matched_version"] = version
                sources.setdefault(source.source_id, payload)
        except Exception as exc:
            errors.append(
                {
                    "minecraft_version": version,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "schema_version": "mmm/forced-project-rag-query-v1",
        "sources": [sources[key] for key in sorted(sources)],
        "errors": errors,
    }


def _existing_code_index() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path("rag/project-index.json"))
    workspace = os.environ.get("MMM_WORKSPACE", "").strip()
    if workspace:
        candidates.append(Path(workspace).expanduser() / "rag/project-index.json")
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _search_code_index(index_path: Path | None, query: str) -> dict[str, Any]:
    if index_path is None:
        return {
            "schema_version": "mmm/forced-code-rag-query-v1",
            "status": "not_indexed",
            "hits": [],
        }
    try:
        result = ProjectRAGIndex(index_path).search_with_receipt(
            query,
            limit=8,
            semantic=False,
            rerank=False,
        )
        return {
            "schema_version": "mmm/forced-code-rag-query-v1",
            "status": "searched",
            "hits": [asdict(hit) for hit in result.hits],
            "receipt": asdict(result.receipt),
        }
    except Exception as exc:
        return {
            "schema_version": "mmm/forced-code-rag-query-v1",
            "status": "error",
            "hits": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "research_progress_snapshot",
    "set_research_progress_hook",
]
