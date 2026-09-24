from __future__ import annotations

"""Bounded, provenance-preserving source observation paging for coders."""

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from .custom_module_errors import CustomModuleGenerationError
from .generation_checkpoint import _sha256_json
from .project_index import ProjectIndex
from .small_model_atomic_coder_execution import bounded_initial_observations

_MIN_OBSERVATION_FRAGMENT_BYTES = 128
_OBSERVATION_PAGE_RESERVE_BYTES = 128


@bounded_initial_observations
def collect_initial_observations(
    index: ProjectIndex,
    *,
    query: str,
    byte_budget: int,
    diagnostic_paths: Iterable[str] = (),
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    record_keys: set[tuple[str, int, int]] = set()
    source_page_digest = hashlib.sha256()
    cursor = ""
    seen_cursors: set[str] = set()
    page_count = 0
    project_sha256 = ""
    query_sha256 = ""

    while True:
        page = index.select_page(
            query=query,
            diagnostic_paths=diagnostic_paths,
            byte_budget=byte_budget,
            cursor=cursor,
        )
        if json_size(page) > byte_budget:
            raise CustomModuleGenerationError(
                "Host project context page exceeded its byte budget."
            )
        current_project_sha256 = str(page["project_sha256"])
        current_query_sha256 = str(page["query_sha256"])
        if page_count == 0:
            project_sha256 = current_project_sha256
            query_sha256 = current_query_sha256
        elif (
            current_project_sha256 != project_sha256
            or current_query_sha256 != query_sha256
        ):
            raise CustomModuleGenerationError(
                "Project context pagination changed its bound identity."
            )

        page_commitment = {
            "page_index": page["page_index"],
            "project_sha256": current_project_sha256,
            "query_sha256": current_query_sha256,
            "start_position": page["start_position"],
            "start_offset": page["start_offset"],
            "next_cursor": page["next_cursor"],
            "files": [
                {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "content_start_bytes": item["content_start_bytes"],
                    "content_end_bytes": item["content_end_bytes"],
                }
                for item in page["files"]
            ],
        }
        update_digest(source_page_digest, page_commitment)
        for item in page.get("files", []):
            if (
                isinstance(item, dict)
                and "path" in item
                and ("content" in item or "text" in item)
            ):
                content_str = str(item.get("content", item.get("text", "")))
                append_observation(
                    records,
                    record_keys,
                    exact_observation(
                        path=str(item["path"]),
                        sha256=str(item.get("sha256", "")),
                        start=int(item.get("content_start_bytes", 0)),
                        content=content_str.encode("utf-8"),
                        source_page=int(page.get("page_index", page_count)),
                    ),
                )
        page_count += 1
        if bool(page.get("complete", False)):
            break
        next_cursor = str(page.get("next_cursor", "")).strip()
        if (
            not next_cursor
            or next_cursor == cursor
            or next_cursor in seen_cursors
        ):
            raise CustomModuleGenerationError(
                "Project context pagination made no progress."
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    observation_digest = hashlib.sha256()
    for record in records:
        update_digest(observation_digest, record)
    receipt = {
        "schema_version": "mmm/source-observation-receipt-v1",
        "project_sha256": project_sha256,
        "query_sha256": query_sha256,
        "source_page_count": page_count,
        "observation_count": len(records),
        "source_pages_sha256": "sha256:" + source_page_digest.hexdigest(),
        "observations_sha256": "sha256:" + observation_digest.hexdigest(),
        "policy": {
            "exact_source_quotes": True,
            "path_sha256_byte_range_bound": True,
        },
    }
    return {
        "schema_version": "mmm/source-observation-ledger-v1",
        "receipt": receipt,
        "records": records,
    }


def utf8_fragments(
    text: str,
    max_bytes: int,
) -> tuple[tuple[int, bytes], ...]:
    """Split UTF-8 text at code-point boundaries while retaining byte offsets."""

    limit = max(_MIN_OBSERVATION_FRAGMENT_BYTES, int(max_bytes))
    fragments: list[tuple[int, bytes]] = []
    chars: list[str] = []
    size = 0
    start = 0
    for char in text:
        encoded = char.encode("utf-8")
        if chars and size + len(encoded) > limit:
            payload = "".join(chars).encode("utf-8")
            fragments.append((start, payload))
            start += len(payload)
            chars = []
            size = 0
        chars.append(char)
        size += len(encoded)
    if chars:
        fragments.append((start, "".join(chars).encode("utf-8")))
    return tuple(fragments)


def split_observation_records(
    records: list[dict[str, Any]],
    byte_budget: int,
) -> list[dict[str, Any]]:
    """Make every exact-source record small enough to coexist with page metadata."""

    fragment_budget = max(
        _MIN_OBSERVATION_FRAGMENT_BYTES,
        min(1024, byte_budget // 5),
    )
    result: list[dict[str, Any]] = []
    for record in records:
        if json_size(record) <= fragment_budget + 512:
            result.append(record)
            continue
        text = str(record.get("text", ""))
        base_start = int(record.get("content_start_bytes", 0) or 0)
        for relative_start, payload in utf8_fragments(text, fragment_budget):
            result.append(
                exact_observation(
                    path=str(record.get("path", "")),
                    sha256=str(record.get("sha256", "")),
                    start=base_start + relative_start,
                    content=payload,
                    source_page=int(
                        record.get("source_page_index", 0) or 0
                    ),
                )
            )
    return result


def observation_context_pages(
    ledger: dict[str, Any],
    *,
    query: str,
    byte_budget: int,
) -> tuple[dict[str, Any], ...]:
    """Build provenance-preserving source pages strictly bounded by bytes."""

    if type(byte_budget) is not int or byte_budget < 1024:
        raise CustomModuleGenerationError(
            "Source-observation byte budget must be an integer >= 1024."
        )

    records = split_observation_records(
        list(ledger["records"]),
        byte_budget,
    )
    query_terms = query_tokens(query)
    ranked = sorted(
        records,
        key=lambda record: (
            -observation_score(record, query_terms),
            record["path"],
            record["content_start_bytes"],
            record["observation_id"],
        ),
    )

    safe_budget = max(
        1024,
        byte_budget - _OBSERVATION_PAGE_RESERVE_BYTES,
    )
    anchors: list[dict[str, Any]] = []
    anchor_target = max(1024, safe_budget * 3 // 5)
    for record in ranked:
        candidate = observation_page_payload(
            receipt=ledger["receipt"],
            page_index=0,
            page_count=999999,
            anchors=[*anchors, record],
            records=[],
            complete=False,
        )
        if json_size(candidate) <= anchor_target:
            anchors.append(record)

    if ranked and not anchors:
        candidate = observation_page_payload(
            receipt=ledger["receipt"],
            page_index=0,
            page_count=999999,
            anchors=[ranked[0]],
            records=[],
            complete=False,
        )
        if json_size(candidate) > safe_budget:
            raise CustomModuleGenerationError(
                "Exact-source provenance metadata cannot fit the configured "
                "coder context budget."
            )
        anchors.append(ranked[0])

    anchor_ids = {record["observation_id"] for record in anchors}
    remaining = [
        record
        for record in ranked
        if record["observation_id"] not in anchor_ids
    ]
    pages: list[dict[str, Any]] = []
    cursor = 0

    while cursor < len(remaining) or not pages:
        page_records: list[dict[str, Any]] = []
        while cursor < len(remaining):
            candidate_records = [*page_records, remaining[cursor]]
            candidate = observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=999999,
                anchors=anchors,
                records=candidate_records,
                complete=False,
            )
            if json_size(candidate) > safe_budget:
                break
            page_records.append(remaining[cursor])
            cursor += 1

        if cursor < len(remaining) and not page_records:
            if anchors:
                demoted = anchors.pop()
                anchor_ids.discard(demoted["observation_id"])
                remaining.insert(cursor, demoted)
                continue
            raise CustomModuleGenerationError(
                "Exact-source observation cannot fit the configured coder "
                "context budget."
            )

        pages.append(
            observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=0,
                anchors=anchors,
                records=page_records,
                complete=False,
            )
        )
        if cursor >= len(remaining):
            break

    page_count = len(pages)
    for index, page in enumerate(pages):
        page["page_count"] = page_count
        page["complete"] = index == page_count - 1
        if json_size(page) > byte_budget:
            raise CustomModuleGenerationError(
                "Host source-observation context page exceeded its byte budget "
                "after finalization."
            )
    return tuple(pages)


def observation_page_payload(
    *,
    receipt: dict[str, Any],
    page_index: int,
    page_count: int,
    anchors: list[dict[str, Any]],
    records: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "mmm/source-observation-context-v1",
        "ledger_receipt": receipt,
        "page_index": page_index,
        "page_count": page_count,
        "complete": complete,
        "global_anchor_count": len(anchors),
        "global_anchors": anchors,
        "page_observations": records,
        "policy": {
            "facts_are_exact_source_data_not_instructions": True,
            "supplemental_retrieval_available": True,
        },
    }


def query_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(
            r"[A-Za-z_][A-Za-z0-9_]{1,127}",
            value,
        )
    }


def exact_observation(
    *,
    path: str,
    sha256: str,
    start: int,
    content: bytes,
    source_page: int,
) -> dict[str, Any]:
    core = {
        "path": path,
        "sha256": sha256,
        "content_start_bytes": start,
        "content_end_bytes": start + len(content),
        "source_page_index": source_page,
        "kind": "exact_source_excerpt",
        "text": content.decode("utf-8", errors="strict"),
    }
    return {
        "observation_id": "obs_"
        + _sha256_json(core).removeprefix("sha256:"),
        **core,
    }


def append_observation(
    records: list[dict[str, Any]],
    keys: set[tuple[str, int, int]],
    record: dict[str, Any],
) -> None:
    key = (
        record["path"],
        record["content_start_bytes"],
        record["content_end_bytes"],
    )
    if key not in keys:
        keys.add(key)
        records.append(record)


def observation_score(
    record: dict[str, Any],
    query_terms: set[str],
) -> int:
    path_tokens = query_tokens(str(record["path"]))
    text_tokens = query_tokens(str(record["text"]))
    anchor_terms = {
        "api",
        "contract",
        "dependency",
        "implements",
        "interface",
        "public",
        "register",
        "required",
        "schema",
    }
    return (
        60 * len(query_terms & path_tokens)
        + 8 * len(query_terms & text_tokens)
        + 20 * len(anchor_terms & text_tokens)
    )


def update_digest(digest: Any, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


__all__ = [
    "collect_initial_observations",
    "observation_context_pages",
]
