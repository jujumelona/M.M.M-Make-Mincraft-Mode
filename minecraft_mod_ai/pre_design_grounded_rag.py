from __future__ import annotations

"""Transport and evidence storage for host-owned Minecraft research.

Retrieval policy intentionally lives in ``catalog_first_grounded_rag``. This module owns
provider I/O and exact evidence materialization only. CurseForge browsing
is intentionally bounded to multiple catalogue windows; receipts identify these
samples as incomplete. Other providers retain their own retrieval budgets.
"""

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from threading import Lock
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .deadline_executor import iter_completed_with_deadlines
from .knowledge import (
    AuthoritativeEvidenceRetriever,
    evidence_catalog_for_version,
    target_neutral_evidence_catalog,
)
from .rag_index import ProjectRAGIndex
from .source_intent_aliases import source_intent_terms, source_query_variants

_TIMEOUT = 8.0
_UA = "MMM-PreDesignResearch/3.0 (+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)"


def _positive_env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _worker_count(task_count: int, env_name: str) -> int:
    """Use all independent tasks unless the operator explicitly constrains fan-out."""
    count = max(0, int(task_count))
    if count <= 1:
        return count
    configured = _positive_env_int(env_name)
    return min(count, configured) if configured is not None else count


def _query_worker_count(task_count: int) -> int:
    return _worker_count(task_count, "MMM_PREDESIGN_QUERY_WORKERS")


def _source_worker_count(task_count: int) -> int:
    return _worker_count(task_count, "MMM_PREDESIGN_SOURCE_WORKERS")


def _provider_result_limit() -> int | None:
    return _positive_env_int("MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY")


def _provider_page_limit() -> int | None:
    return _positive_env_int("MMM_PREDESIGN_PROVIDER_SEARCH_PAGES")


def _limit_reached(value: int, limit: int | None) -> bool:
    return limit is not None and value >= limit


def _search_terms(value: str) -> tuple[str, ...]:
    stop = {
        "minecraft",
        "fabric",
        "mod",
        "mods",
        "game",
        "system",
        "feature",
        "implementation",
        "source",
    }
    words: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_+.#/-]+|[가-힣]{2,}", value):
        key = token.casefold()
        if len(key) < 3 or key in stop or key in words:
            continue
        words.append(key)
    return tuple(words)


def _query_terms(value: str) -> str:
    words = _search_terms(value)
    return " ".join(words) or "minecraft fabric"


def _semantic_page_size(
    query: str,
    provider_max: int,
    result_limit: int | None,
    current: int,
) -> int:
    """Use provider pagination capacity; query word count is not a recall budget."""
    remaining = None if result_limit is None else max(0, result_limit - current)
    if remaining == 0:
        return 0
    page_size = provider_max
    return page_size if remaining is None else min(page_size, remaining)


def _coverage_gain(
    wanted: set[str],
    covered: set[str],
    values: Sequence[Any],
) -> int:
    if not wanted:
        return 1
    observed: set[str] = set()
    for value in values:
        observed.update(_search_terms(str(value or "")))
    before = len(covered)
    covered.update(wanted & observed)
    return len(covered) - before


def _json(url: str, headers: Mapping[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": _UA,
            **dict(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _text(url: str, headers: Mapping[str, str] | None = None) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": _UA,
            **dict(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def _error(provider: str, exc: BaseException) -> dict[str, Any]:
    code = getattr(exc, "code", None)
    return {
        "provider": provider,
        "status": "error",
        "http_status": code if isinstance(code, int) else None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _versions(router: Any) -> tuple[str, ...]:
    requested = str(
        getattr(router, "_mmm_requested_minecraft_version", "") or ""
    ).strip()
    existing = str(
        getattr(router, "_mmm_existing_minecraft_version", "") or ""
    ).strip()
    return (requested,) if requested else ((existing,) if existing else ())


def _search_authoritative_catalog(
    query: str, versions: tuple[str, ...]
) -> dict[str, Any]:
    retriever = AuthoritativeEvidenceRetriever()
    records: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for version in versions or ("",):
        try:
            catalog = (
                evidence_catalog_for_version(version)
                if version
                else target_neutral_evidence_catalog()
            )
            kwargs = {"minecraft_version": version} if version else {}
            for source in retriever.search(query, limit=len(catalog), **kwargs):
                item = asdict(source)
                item["matched_version"] = version
                records.setdefault(source.source_id, item)
        except Exception as exc:
            errors.append(
                {
                    "minecraft_version": version,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "schema_version": "mmm/project-rag-query-v3",
        "sources": list(records.values()),
        "errors": errors,
    }


def _existing_code_index() -> Path | None:
    values = [os.environ.get("MMM_PROJECT_RAG_INDEX", ""), "rag/project-index.json"]
    if os.environ.get("MMM_WORKSPACE"):
        values.append(str(Path(os.environ["MMM_WORKSPACE"]) / "rag/project-index.json"))
    for raw in values:
        if not str(raw).strip():
            continue
        path = Path(str(raw)).expanduser().resolve()
        if path.is_file():
            return path
    return None


def _search_code_index(index: Path | None, query: str) -> dict[str, Any]:
    if index is None:
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "not_indexed",
            "hits": [],
        }
    try:
        result = ProjectRAGIndex(index).search_with_receipt(
            query,
            limit=max(1, len(_search_terms(query))),
            semantic=False,
            rerank=False,
        )
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "searched",
            "hits": [asdict(hit) for hit in result.hits],
            "receipt": asdict(result.receipt),
        }
    except Exception as exc:
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "error",
            "hits": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _body(record: Mapping[str, Any]) -> str:
    return str(
        record.get("content") or record.get("body") or record.get("text") or ""
    ).strip()


def _units(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    grounded = (
        evidence.get("grounded_rag")
        if isinstance(evidence.get("grounded_rag"), Mapping)
        else {}
    )
    result: list[dict[str, Any]] = []
    seen_bodies: set[str] = set()

    for row in (
        grounded.get("queries", [])
        if isinstance(grounded.get("queries"), list)
        else []
    ):
        if not isinstance(row, Mapping):
            continue
        for record in (
            row.get("evidence_records", [])
            if isinstance(row.get("evidence_records"), list)
            else []
        ):
            if not isinstance(record, Mapping):
                continue
            body = _body(record)
            if not body:
                continue
            body_key = str(record.get("content_sha256") or "").strip() or _sha256_text(body)
            if body_key in seen_bodies:
                continue
            seen_bodies.add(body_key)
            result.append(
                {
                    "query": str(row.get("query") or ""),
                    "source_id": str(record.get("source_id") or ""),
                    "source_type": str(record.get("source_type") or ""),
                    "url": str(record.get("url") or ""),
                    "title": str(record.get("title") or ""),
                    "content_sha256": str(
                        record.get("content_sha256") or _sha256_text(body)
                    ),
                    "content": body,
                }
            )
    return result


def _root() -> Path:
    configured = os.environ.get("MMM_RESEARCH_DOCUMENT_DIR", "").strip()
    workspace = os.environ.get("MMM_WORKSPACE", "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else (Path(workspace).expanduser() if workspace else Path.cwd())
        / "mmm-output"
        / "research-evidence"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    try:
        if path.is_file() and path.read_bytes() == encoded:
            return
    except OSError:
        pass

    name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            name = handle.name

        for attempt in range(4):
            try:
                os.replace(name, path)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if name:
            Path(name).unlink(missing_ok=True)


def _build_evidence_pages(
    domain_id: str,
    evidence: Mapping[str, Any],
    digest: str,
) -> tuple[list[dict[str, Any]], str]:
    units = _units(evidence)
    pages: list[dict[str, Any]] = []

    for unit_index, unit in enumerate(units):
        rendered = json.dumps(
            unit, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        pages.append(
            {
                "schema_version": "mmm/research-evidence-page-v2",
                "domain_id": domain_id,
                "unit_id": f"source:{unit_index}",
                "part_index": 0,
                "part_count": 1,
                "content": rendered,
            }
        )

    total_pages = len(pages)
    for index, page in enumerate(pages):
        page.update(
            {
                "page_index": index,
                "page_count": total_pages,
                "page_ref": f"{digest}#page={index + 1}/{total_pages}",
            }
        )

    page_lines: list[str] = []
    for page in pages:
        rendered_page = json.dumps(
            page,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        clean_page = (
            rendered_page.replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
            .replace("\x85", "\\u0085")
        )
        page_lines.append(clean_page)

    serialized = "".join(f"{line}\n" for line in page_lines)
    return pages, serialized


def _materialize_domain_evidence_document(
    domain_id: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    raw = json.dumps(dict(evidence), ensure_ascii=False, sort_keys=True, default=str)
    digest = _sha256_text(raw)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain_id).strip("_") or "unknown"
    root = _root()
    raw_path = root / f"{safe}-{digest[7:19]}.json"
    pages_path = root / f"{safe}-{digest[7:19]}.pages.jsonl"

    pages, pages_text = _build_evidence_pages(domain_id, evidence, digest)
    _write(raw_path, raw)
    _write(pages_path, pages_text)

    return {
        "schema_version": "mmm/research-evidence-document-v2",
        "domain_id": domain_id,
        "document_sha256": digest,
        "raw_path": str(raw_path),
        "pages_path": str(pages_path),
        "page_count": len(pages),
        "page_partition": "claim_bearing_source_unit",
        "source_keys": sorted(str(key) for key in evidence),
        "model_unit_count": len(pages),
        "model_projection": "claim_bearing_source_bodies_only",
        "_pages": pages,
    }


def _read_evidence_pages(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = int(document.get("page_count") or 0)
    if expected == 0:
        return []

    cached = document.get("_pages")
    if isinstance(cached, list) and len(cached) == expected:
        return [dict(page) for page in cached if isinstance(page, Mapping)]

    path = Path(str(document.get("pages_path") or "")).expanduser()
    raw_path = Path(str(document.get("raw_path") or "")).expanduser()
    pages: list[dict[str, Any]] = []
    read_ok = False

    if path.is_file():
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stripped = line.strip("\r\n")
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    if isinstance(record, Mapping):
                        pages.append(dict(record))
            if len(pages) == expected:
                read_ok = True
        except (OSError, json.JSONDecodeError, ValueError):
            pages = []
            read_ok = False

    if not read_ok and raw_path.is_file():
        try:
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
            evidence_dict = json.loads(raw_text)
            if isinstance(evidence_dict, Mapping):
                domain_id = str(document.get("domain_id") or "")
                digest = str(document.get("document_sha256") or "") or _sha256_text(raw_text)
                reconstructed, pages_text = _build_evidence_pages(
                    domain_id, evidence_dict, digest
                )
                _write(path, pages_text)
                pages = reconstructed
                read_ok = True
        except Exception:
            pass

    if not read_ok and not pages:
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
        except OSError:
            pass
        return []

    return pages


def _prompt_document_receipt(document: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "domain_id",
        "document_sha256",
        "page_count",
        "page_partition",
        "source_keys",
        "model_unit_count",
        "model_projection",
    )
    return {key: document[key] for key in keep if key in document}


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


__all__ = [
    "_error",
    "_existing_code_index",
    "_materialize_domain_evidence_document",
    "_prompt_document_receipt",
    "_query_worker_count",
    "_read_evidence_pages",
    "_search_authoritative_catalog",
    "_search_code_index",
    "_sha256",
    "_sha256_text",
    "_source_worker_count",
    "_versions",
]
