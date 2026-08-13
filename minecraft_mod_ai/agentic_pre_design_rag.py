from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .knowledge import AuthoritativeEvidenceRetriever, evidence_catalog_for_version
from .rag_index import ProjectRAGIndex


_MARKER = "_mmm_forced_pre_design_rag_v3"
_FORCED_RAG_CONTEXT: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "mmm_forced_pre_design_rag_context",
    default=None,
)
_EVIDENCE_PAGE_CHARS = 12_000
_EVIDENCE_DOCUMENT_SCHEMA = "mmm/research-evidence-document-v1"
_EVIDENCE_PAGE_SCHEMA = "mmm/research-evidence-page-v1"


def harden_pre_design_research(agentic_module: Any) -> None:
    """Force deterministic RAG and feed model workers through bounded evidence documents.

    Full retrieval receipts remain authoritative and are retained in the returned research
    bundle. Prompt-facing workers never receive those raw receipts directly: the host writes
    them to a durable per-run document, reads every bounded page, asks the planner to digest
    each page separately, then synthesizes only the compact page notes. This keeps fixed
    context limits independent of retrieval volume without reducing any research route.
    """

    current_collect = agentic_module.collect_pre_design_research
    if getattr(current_collect, _MARKER, False):
        return

    # Preserve the central intelligence parallel collector when it already owns provider/
    # domain fan-out. Only obsolete forced-RAG wrappers are unwrapped.
    if getattr(current_collect, "_mmm_parallel_research_design_core_v1", False):
        original_collect = current_collect
    else:
        original_collect = getattr(current_collect, "__wrapped__", current_collect)
    current_slice = agentic_module._domain_evidence_slice
    original_domain_slice = getattr(current_slice, "__wrapped__", current_slice)
    original_domain_worker = agentic_module._research_domain_with_agent

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

        pages = _read_evidence_pages(document)
        page_notes: list[dict[str, Any]] = []
        for page in pages:
            raw = router.generate_text(
                "planner",
                _research_page_messages(
                    prompt=prompt,
                    domain=domain,
                    document=document,
                    page=page,
                ),
                response_format="json",
                response_schema=agentic_module._RESEARCH_NOTE_SCHEMA,
                tool_stage="research",
                enable_tools=False,
            )
            note = agentic_module._parse_research_note(raw, domain_id)
            page_ref = str(page.get("page_ref", "")).strip()
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
            page_notes.append(
                {
                    "page_ref": page_ref,
                    "unit_id": str(page.get("unit_id", "")),
                    "claims": claims,
                    "gaps": list(note.get("gaps", [])),
                    "next_queries": list(note.get("next_queries", [])),
                }
            )

        synthesis_evidence = {
            "evidence_document": _prompt_document_receipt(document),
            "page_notes": page_notes,
        }
        trace = agentic_module.PlannerStageTrace(
            stage="pre_design_research",
            prompt=prompt,
            metadata={"domain_id": domain_id, **dict(trace_metadata or {})},
        )
        seen: set[str] = set()
        prior: dict[str, Any] | None = None

        while True:
            messages = agentic_module._research_messages(
                prompt=prompt,
                domain=domain,
                deterministic_evidence=synthesis_evidence,
                prior=prior,
            )
            raw = router.generate_text(
                "planner",
                messages,
                response_format="json",
                response_schema=agentic_module._RESEARCH_NOTE_SCHEMA,
                tool_stage="research",
                enable_tools=True,
            )
            try:
                note = agentic_module._parse_research_note(raw, domain_id)
            except agentic_module.SpecValidationError as exc:
                state = agentic_module._json_sha256(
                    {"error": str(exc), "raw": raw.strip()}
                )
                trace.record_attempt(
                    raw_output=raw,
                    validation_error=str(exc),
                    candidate=None,
                    context={"domain_id": domain_id},
                )
                if state in seen:
                    return {
                        "domain_id": domain_id,
                        "claims": [],
                        "gaps": [str(exc)],
                        "next_queries": [],
                        "sufficient": False,
                        "fixed_point": True,
                        "evidence_document": _prompt_document_receipt(document),
                    }
                seen.add(state)
                prior = {
                    "domain_id": domain_id,
                    "claims": [],
                    "gaps": [str(exc)],
                    "next_queries": list(domain.get("queries", [])),
                    "sufficient": False,
                }
                continue

            trace.record_attempt(
                raw_output=raw,
                validation_error=None,
                candidate=note,
                accepted=note if note["sufficient"] else None,
                context={"domain_id": domain_id},
            )
            state = agentic_module._json_sha256(note)
            if note["sufficient"]:
                trace.record_success(note)
                return {
                    **note,
                    "evidence_document": _prompt_document_receipt(document),
                }
            if state in seen:
                return {
                    **note,
                    "fixed_point": True,
                    "evidence_document": _prompt_document_receipt(document),
                }
            seen.add(state)
            prior = note

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

    setattr(collect, _MARKER, True)
    collect._mmm_forced_pre_design_rag_v1 = True  # type: ignore[attr-defined]
    collect._mmm_forced_pre_design_rag_v2 = True  # type: ignore[attr-defined]
    collect.__wrapped__ = original_collect  # type: ignore[attr-defined]
    domain_slice.__wrapped__ = original_domain_slice  # type: ignore[attr-defined]
    research_domain_from_document.__wrapped__ = original_domain_worker  # type: ignore[attr-defined]
    research_domain_from_document._mmm_document_paged_evidence_v1 = True  # type: ignore[attr-defined]
    agentic_module.collect_pre_design_research = collect
    agentic_module._domain_evidence_slice = domain_slice
    agentic_module._research_domain_with_agent = research_domain_from_document
    agentic_module._research_receipt = compact_receipt


def _research_page_messages(
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    page: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Create one bounded page-reading request; raw cross-page evidence is never inlined."""
    system = (
        "You are reading exactly one bounded page from a host-owned Minecraft research "
        "evidence document. Extract only design-relevant claims supported by this page. "
        "Do not assume unseen pages are absent; the host will read every page and synthesize "
        "the page notes later. Return one compact JSON object matching research_note. "
        "research_note.domain_id must equal the assigned domain. Evidence refs should use "
        "the supplied page_ref. Set sufficient=true when this page has been fully processed; "
        "it does not mean the whole domain is complete."
    )
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
            "content": page.get("content", ""),
        },
        "instruction": (
            "Read the complete supplied page. Preserve source identifiers and concrete "
            "version/API facts. Put unresolved page-local uncertainty in gaps."
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


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
    pages: list[dict[str, Any]] = []
    for unit_id, value in units:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        parts = [
            rendered[offset : offset + _EVIDENCE_PAGE_CHARS]
            for offset in range(0, len(rendered), _EVIDENCE_PAGE_CHARS)
        ] or [""]
        for part_index, content in enumerate(parts):
            pages.append(
                {
                    "schema_version": _EVIDENCE_PAGE_SCHEMA,
                    "domain_id": domain_id,
                    "unit_id": unit_id,
                    "part_index": part_index,
                    "part_count": len(parts),
                    "content": content,
                }
            )

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
        "source_keys": sorted(str(key) for key in evidence),
    }


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
            if len(content) > _EVIDENCE_PAGE_CHARS:
                raise ValueError(
                    f"Research evidence page exceeds {_EVIDENCE_PAGE_CHARS} chars: {path}"
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
    return ("1.20.1", "1.21.1")


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