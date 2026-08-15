from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .knowledge import AuthoritativeEvidenceRetriever, evidence_catalog_for_version
from .rag_index import ProjectRAGIndex


_MARKER = "_mmm_forced_pre_design_rag_v3"
_SLICE_OWNER_MARKER = "_mmm_forced_rag_slice_owner_v1"
_WORKER_OWNER_MARKER = "_mmm_document_worker_owner_v1"
_FORCED_RAG_CONTEXT: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "mmm_forced_pre_design_rag_context",
    default=None,
)
_EVIDENCE_PAGE_CHARS = 1_800
_EVIDENCE_DOCUMENT_SCHEMA = "mmm/research-evidence-document-v1"
_EVIDENCE_PAGE_SCHEMA = "mmm/research-evidence-page-v1"
_DOMAIN_CHECKPOINT_SCHEMA = "mmm/research-domain-checkpoint-v2"
_PAGE_PROTOCOL_SCHEMA = "mmm/research-page-continuation-v2"
_SYNTHESIS_PROTOCOL_SCHEMA = "mmm/research-hierarchical-synthesis-v2"
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
        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if callable(hook):
        try:
            hook(dict(payload))
        except Exception:
            # Observability must never change research execution semantics.
            pass


def _effective_forced_collect_owner(current: Any) -> bool:
    """Return true only when forced RAG wraps (rather than sits behind) fan-out."""

    saw_parallel_owner = False
    seen: set[int] = set()
    cursor = current
    while callable(cursor) and id(cursor) not in seen:
        seen.add(id(cursor))
        if cursor.__dict__.get(_MARKER) is cursor:
            return not saw_parallel_owner
        if (
            cursor.__dict__.get("_mmm_parallel_research_design_core_v1")
            is cursor
        ):
            saw_parallel_owner = True
        cursor = getattr(cursor, "__wrapped__", None)
    return False


def harden_pre_design_research(agentic_module: Any) -> None:
    """Force deterministic RAG and feed model workers through bounded evidence documents.

    Full retrieval receipts remain authoritative and are retained in the returned research
    bundle. Prompt-facing workers never receive those raw receipts directly: the host writes
    them to a durable per-run document, reads every bounded page, asks the planner to digest
    each page separately, then synthesizes only the compact page notes. This keeps fixed
    context limits independent of retrieval volume without reducing any research route.
    """

    current_collect = agentic_module.collect_pre_design_research
    # functools.wraps copies function attributes. An outer parallel collector can
    # therefore inherit our boolean marker without actually executing this wrapper.
    # Identity ownership distinguishes the real forced-RAG owner from copied metadata.
    if _effective_forced_collect_owner(current_collect):
        return

    # Preserve the central intelligence parallel collector when it already owns provider/
    # domain fan-out. Only obsolete forced-RAG wrappers are unwrapped.
    if getattr(current_collect, "_mmm_parallel_research_design_core_v1", False):
        original_collect = current_collect
    else:
        original_collect = getattr(current_collect, "__wrapped__", current_collect)
    current_slice = agentic_module._domain_evidence_slice
    slice_is_owned = (
        current_slice.__dict__.get(_SLICE_OWNER_MARKER) is current_slice
    )
    original_domain_slice = getattr(current_slice, "__wrapped__", current_slice)
    current_domain_worker = agentic_module._research_domain_with_agent
    worker_is_owned = (
        current_domain_worker.__dict__.get(_WORKER_OWNER_MARKER)
        is current_domain_worker
    )
    original_domain_worker = current_domain_worker

    def collect(
        router: Any,
        prompt: str,
        *,
        trace_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        brief = agentic_module.normalize_research_brief(
            prompt,
            {"title": "pre-design research"},
        )
        forced = _forced_rag_bundle(router, brief)
        token = _FORCED_RAG_CONTEXT.set(forced)
        try:
            result = original_collect(
                router,
                prompt,
                trace_metadata=trace_metadata,
            )
        finally:
            _FORCED_RAG_CONTEXT.reset(token)

        deterministic = result.get("deterministic")
        if not isinstance(deterministic, dict):
            deterministic = {}
        result = {
            **result,
            "deterministic": {
                **deterministic,
                "forced_project_rag": forced,
            },
        }
        result["research_sha256"] = _sha256(result)
        return result

    def domain_slice(domain_id: str, deterministic: Mapping[str, Any]) -> dict[str, Any]:
        # Build the complete raw evidence slice first. It is persisted verbatim below, but
        # never returned inline to the model-facing prompt path.
        raw_value = dict(original_domain_slice(domain_id, deterministic))
        forced = _FORCED_RAG_CONTEXT.get()
        if not isinstance(forced, Mapping):
            forced = deterministic.get("forced_project_rag")
        if isinstance(forced, Mapping):
            domains = forced.get("domains")
            selected = None
            if isinstance(domains, list):
                selected = next(
                    (
                        item
                        for item in domains
                        if isinstance(item, Mapping)
                        and item.get("domain_id") == domain_id
                    ),
                    None,
                )
            receipt = {key: item for key, item in forced.items() if key != "domains"}
            if isinstance(selected, Mapping):
                receipt.update(dict(selected))
            raw_value["forced_project_rag"] = receipt

        document = _materialize_domain_evidence_document(domain_id, raw_value)
        return {"evidence_document": document}

    def research_domain_from_document(
        router: Any,
        *,
        prompt: str,
        domain: Mapping[str, Any],
        deterministic: Mapping[str, Any],
        trace_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
        evidence = agentic_module._domain_evidence_slice(domain_id, deterministic)
        document = evidence.get("evidence_document")
        if not isinstance(document, Mapping):
            return original_domain_worker(
                router,
                prompt=prompt,
                domain=domain,
                deterministic=deterministic,
                trace_metadata=trace_metadata,
            )

        return _research_document_domain(
            agentic_module,
            router,
            prompt=prompt,
            domain=domain,
            document=document,
            trace_metadata=trace_metadata,
        )

    def compact_receipt(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        keep = (
            "schema_version",
            "research_sha256",
            "evidence_sha256",
            "radar_sha256",
            "route_sha256",
            "query_sha256",
            "status",
            "unresolved_official_domains",
            "candidate_count",
            "domain_count",
            "query_count",
            "project_source_count",
            "code_index_status",
            "code_index_path",
            "document_sha256",
            "page_count",
        )
        return {key: value[key] for key in keep if key in value}

    setattr(collect, _MARKER, collect)
    collect._mmm_forced_pre_design_rag_v1 = True  # type: ignore[attr-defined]
    collect._mmm_forced_pre_design_rag_v2 = True  # type: ignore[attr-defined]
    collect.__wrapped__ = original_collect  # type: ignore[attr-defined]
    domain_slice.__wrapped__ = original_domain_slice  # type: ignore[attr-defined]
    research_domain_from_document.__wrapped__ = original_domain_worker  # type: ignore[attr-defined]
    research_domain_from_document._mmm_document_paged_evidence_v1 = True  # type: ignore[attr-defined]
    setattr(domain_slice, _SLICE_OWNER_MARKER, domain_slice)
    setattr(
        research_domain_from_document,
        _WORKER_OWNER_MARKER,
        research_domain_from_document,
    )
    agentic_module.collect_pre_design_research = collect
    if not slice_is_owned:
        agentic_module._domain_evidence_slice = domain_slice
    if not worker_is_owned:
        agentic_module._research_domain_with_agent = research_domain_from_document
    agentic_module._research_receipt = compact_receipt


def _research_page_messages(
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    page: Mapping[str, Any],
    continuation_offset: int = 0,
) -> list[dict[str, str]]:
    """Create one bounded page-reading request; raw cross-page evidence is never inlined."""
    system = (
        "You are reading exactly one bounded page from a host-owned Minecraft research "
        "evidence document. Extract only design-relevant claims supported by this page. "
        "Do not assume unseen pages are absent; the host will read every page and synthesize "
        "the page notes later. Return one compact JSON object matching research_note. "
        "research_note.domain_id must equal the assigned domain. Evidence refs should use "
        "the supplied page_ref. This is a lossless continuation protocol: emit at most the "
        "schema allowance, advance next_offset beyond continuation_offset, and set complete "
        "only after inspecting through the final character and echoing tail_sha256. Never "
        "drop remaining claims merely because one response page is full. Set sufficient=true "
        "when the supplied continuation span was processed; it does not mean the whole domain "
        "is complete."
    )
    content = str(page.get("content", ""))
    offset = max(0, min(len(content), int(continuation_offset)))
    tail_sha256 = _sha256_text(content)
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
            "tail_sha256": tail_sha256,
        },
        "continuation_contract": {
            "schema_version": _PAGE_PROTOCOL_SCHEMA,
            "current_offset": offset,
            "complete_requires_next_offset": len(content),
            "complete_requires_tail_sha256": tail_sha256,
        },
        "instruction": (
            "Read the complete supplied page. Preserve source identifiers and concrete "
            "version/API facts. Put unresolved page-local uncertainty in gaps. If more "
            "claims remain than fit this response, set complete=false and continue from a "
            "strictly larger next_offset on the next host request."
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


class _BoundedResearchOutputError(RuntimeError):
    pass


def _page_response_schema(research_note_schema: Mapping[str, Any]) -> dict[str, Any]:
    schema = json.loads(json.dumps(dict(research_note_schema)))
    schema["properties"]["continuation"] = {
        "type": "object",
        "properties": {
            "complete": {"type": "boolean"},
            "next_offset": {"type": "integer", "minimum": 0},
            "tail_sha256": {"type": "string", "minLength": 1, "maxLength": 71},
        },
        "required": ["complete", "next_offset", "tail_sha256"],
        "additionalProperties": False,
    }
    schema["required"] = ["research_note", "continuation"]
    return schema


def _core_note(note: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: note[key]
        for key in ("domain_id", "claims", "gaps", "next_queries", "sufficient")
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
    payload = agentic_module._extract_json_object(raw)
    continuation = payload.get("continuation")
    # Focused adapters written before the continuation protocol do not enforce JSON
    # Schema. Preserve that test/adapter compatibility; production schema decoding
    # always requires the explicit receipt below.
    if continuation is None:
        return {
            "note": note,
            "continuation": {
                "complete": True,
                "next_offset": content_chars,
                "tail_sha256": tail_sha256,
                "legacy_adapter": True,
            },
        }
    validated = _validate_continuation_receipt(
        agentic_module,
        continuation,
        current_offset=current_offset,
        content_chars=content_chars,
        tail_sha256=tail_sha256,
    )
    return {"note": note, "continuation": validated}


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
                f"{type(error).__name__}: {str(error)[:500]}. Regenerate from the same "
                "supplied evidence span exactly once. Emit only schema-valid compact JSON; "
                "do not repeat, quote, or continue the invalid output."
            ),
        },
    ]


def _generate_bounded(
    agentic_module: Any,
    router: Any,
    *,
    messages: list[dict[str, str]],
    response_schema: Mapping[str, Any],
    parser: Any,
    progress_label: str,
) -> Any:
    _emit_research_progress("model_attempt", label=progress_label, attempt=1)
    try:
        raw = router.generate_text(
            "planner",
            messages,
            response_format="json",
            response_schema=response_schema,
            tool_stage="research",
            enable_tools=False,
        )
        return parser(raw)
    except Exception as first_error:
        if not isinstance(first_error, agentic_module.SpecValidationError) and not (
            _structured_output_failure(first_error)
        ):
            raise
        _emit_research_progress(
            "bounded_json_repair",
            label=progress_label,
            attempt=2,
            error=f"{type(first_error).__name__}: {str(first_error)[:500]}",
        )
        try:
            repaired = router.generate_text(
                "planner",
                _repair_messages(messages, error=first_error),
                response_format="json",
                response_schema=response_schema,
                tool_stage="research",
                enable_tools=False,
            )
            return parser(repaired)
        except Exception as second_error:
            if not isinstance(
                second_error, agentic_module.SpecValidationError
            ) and not _structured_output_failure(second_error):
                raise
            raise _BoundedResearchOutputError(
                f"bounded JSON repair failed: {second_error}"
            ) from second_error


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
            root = Path(output_root).expanduser() / "research-checkpoints-v2"
        elif workspace:
            workspace_path = Path(workspace).expanduser().resolve()
            if (workspace_path / ".git").exists():
                root = workspace_path.parent / "mmm-output" / "research-checkpoints-v2"
            else:
                root = workspace_path / "research-checkpoints-v2"
        else:
            root = Path(tempfile.gettempdir()) / "mmm-research-checkpoints-v2"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _checkpoint_dir(domain_key: str) -> Path:
    path = _checkpoint_root() / "domains-v2" / domain_key[:2] / domain_key
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
    path = _manifest_path(domain_key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("status") not in {
        "complete",
        "terminal_gap",
    }:
        return None
    note = payload.get("note")
    if (
        payload.get("schema_version") != _DOMAIN_CHECKPOINT_SCHEMA
        or payload.get("domain_key") != domain_key
        or payload.get("note_sha256") != _sha256(note)
    ):
        return None
    try:
        _validate_core_note(agentic_module, note, domain_id)
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
        claims.append(
            {
                "claim": str(claim.get("claim", "")).strip(),
                "evidence_refs": refs,
            }
        )
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
                        "error": f"{type(exc).__name__}: {str(exc)[:600]}",
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
            json.dumps(
                note, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
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
    # Never force two large children past the input bound. Once leaf evidence pages
    # have each been summarized, atomize oversized summaries into bounded semantic
    # records so the next level makes deterministic progress without dropping fields.
    if len(notes) > 1 and len(groups) == len(notes):
        atomic: list[dict[str, Any]] = []
        for note in notes:
            domain_id = str(note.get("domain_id", "unknown"))
            for claim in note.get("claims", []):
                atomic.append(
                    {
                        "domain_id": domain_id,
                        "claims": [claim],
                        "gaps": [],
                        "next_queries": [],
                        "sufficient": bool(note.get("sufficient")),
                    }
                )
            for gap in note.get("gaps", []):
                atomic.append(
                    {
                        "domain_id": domain_id,
                        "claims": [],
                        "gaps": [gap],
                        "next_queries": [],
                        "sufficient": bool(note.get("sufficient")),
                    }
                )
            for query in note.get("next_queries", []):
                atomic.append(
                    {
                        "domain_id": domain_id,
                        "claims": [],
                        "gaps": [],
                        "next_queries": [query],
                        "sufficient": bool(note.get("sufficient")),
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
            "concrete contradictions and gaps. The full lossless claim catalog is retained "
            "by the host, so this response is a bounded design summary, not a replacement."
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a bounded hierarchical Minecraft research synthesizer. Return "
                "only one JSON object matching research_note. Do not use tools and do not "
                "repeat raw evidence."
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
                "sample": [
                    _bounded_domain_projection(item) for item in value[:2]
                ],
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
        except Exception:
            note = None
    if note is not None:
        _emit_research_progress(
            "synthesis_checkpoint_hit",
            domain_id=domain_id,
            level=level,
            group=group_label,
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
        failures.append(
            {
                "unit": f"synthesis:{level}:{group_label}",
                "error": f"{type(exc).__name__}: {str(exc)[:600]}",
            }
        )
        return [
            {
                "domain_id": domain_id,
                "claims": [],
                "gaps": ["A bounded synthesis page failed validation."],
                "next_queries": [],
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
            # Every bounded retry/split at this frontier failed to reduce the work.
            # Stop in host code with an explicit terminal gap instead of reissuing the
            # same malformed synthesis forever or discarding the durable evidence ledger.
            return {
                "domain_id": domain_id,
                "claims": [],
                "gaps": [
                    "Bounded research synthesis reached a validated no-progress frontier; "
                    "full evidence remains in the durable ledger."
                ],
                "next_queries": [],
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
            )
            return cached

        pages = _read_evidence_pages(document)
        failures: list[dict[str, str]] = []
        page_notes: list[dict[str, Any]] = []
        _emit_research_progress(
            "domain_start",
            domain_id=domain_id,
            page_count=len(pages),
        )
        for page_index, page in enumerate(pages):
            _emit_research_progress(
                "page_start",
                domain_id=domain_id,
                page_index=page_index + 1,
                page_count=len(pages),
            )
            # Raw evidence is already a host-owned, hash-addressed page. Preserve every
            # byte in the ledger and let the model synthesize packed page fragments once;
            # forcing a separate model paraphrase before synthesis doubled simple-request
            # latency without adding evidence.
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
        evidence_ledger = _materialize_evidence_ledger(
            domain_key, domain_id, pages
        )
        note: dict[str, Any] = {
            **_core_note(summary),
            "evidence_document": _prompt_document_receipt(document),
            "claim_catalog": catalog,
            "evidence_ledger": evidence_ledger,
            "checkpoint": {
                "schema_version": _DOMAIN_CHECKPOINT_SCHEMA,
                "request_sha256": "sha256:" + domain_key,
                "status": "terminal_gap" if failures else "complete",
            },
        }
        if failures:
            existing_gaps = [str(item) for item in note.get("gaps", [])]
            note["gaps"] = (
                existing_gaps
                + [f"{item['unit']}: {item['error']}" for item in failures]
            )[:4]
            note["sufficient"] = False
            note["research_failures"] = failures
            note["fixed_point"] = True
        elif not bool(note.get("sufficient")):
            note["fixed_point"] = True

        status = "terminal_gap" if failures else "complete"
        _write_manifest(
            domain_key,
            status=status,
            note=note,
            failures=failures,
        )
        _emit_research_progress(
            "domain_complete" if status == "complete" else "domain_gap_receipt",
            domain_id=domain_id,
            status=status,
            claim_count=catalog["claim_count"],
            page_count=len(pages),
            failure_count=len(failures),
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

    # Pack small records, but split oversized records into ordered UTF-8-safe fragments.
    # Concatenating an oversized record's fragments reproduces its exact JSON text; no
    # middle evidence is replaced by a head/tail digest.
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
        "model_projection": "lossless_ordered_utf8_fragments",
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
    for source_key, value in evidence.items():
        if isinstance(value, Mapping):
            queries = value.get("queries")
            if isinstance(queries, list):
                metadata = {key: item for key, item in value.items() if key != "queries"}
                yield f"{source_key}:metadata", metadata
                for index, query in enumerate(queries):
                    yield f"{source_key}:query:{index}", query
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


__all__ = ["harden_pre_design_research"]
