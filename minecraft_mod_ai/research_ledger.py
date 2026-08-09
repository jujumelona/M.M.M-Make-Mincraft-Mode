from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .complete_spec import ProductionModule
INTEGRATION_TYPE = "mmm_research_shard"
_SCHEMA_VERSION = "mmm/research-shard-v1"
_ALLOWED_ROOT = Path(".minecraft_ai/research")
_TOKEN = re.compile(r"[\w][\w.:+/-]{1,127}", re.UNICODE)
_SUBTOKEN = re.compile(r"[\w]+(?:-[\w]+)*", re.UNICODE)
_TECHNICAL_HIT_PATH = re.compile(
    r"^/(primary/hit_ids/\d+|corrections/\d+/hit_ids/\d+)(?:/|$)"
)
_CONTEXT_SCHEMA = "mmm/module-research-context-v1"
_DEFAULT_CONTEXT_BYTES = 8 * 1024


class ResearchLedgerError(RuntimeError):
    pass


def is_research_shard(module: ProductionModule) -> bool:
    return (
        module.kind == "integration"
        and module.config.get("integration_type") == INTEGRATION_TYPE
    )


def write_research_shard(
    project_root: str | Path,
    *,
    module: ProductionModule,
) -> dict[str, Any]:
    """Write one approved research page without invoking a model.

    The ledger is build metadata, not a Fabric feature or packaged resource.
    Existing identical data is accepted for crash-safe resume; conflicting data
    fails closed.
    """

    module.validate()
    if not is_research_shard(module):
        raise ResearchLedgerError("Module is not an MMM research shard.")
    config = module.config
    if config.get("schema_version") != _SCHEMA_VERSION:
        raise ResearchLedgerError("Research shard schema_version is invalid.")
    artifact = config.get("artifact")
    if not isinstance(artifact, dict):
        raise ResearchLedgerError("Research shard artifact contract is missing.")
    if artifact.get("write_mode") != "exact_json_resource_only":
        raise ResearchLedgerError("Research shard write_mode is invalid.")
    if artifact.get("generate_java_or_gameplay_feature") is not False:
        raise ResearchLedgerError(
            "Research shards may not authorize Java or gameplay generation."
        )

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ResearchLedgerError("Research ledger target must be a real project.")
    relative = _safe_target_path(artifact.get("target_path"))
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:  # pragma: no cover - guarded by _safe_target_path
        raise ResearchLedgerError("Research ledger path escapes the project.") from exc
    _reject_symlink_parents(root, target)

    body = {
        "schema_version": _SCHEMA_VERSION,
        "module_id": module.module_id,
        "shard_index": config.get("shard_index"),
        "shard_count": config.get("shard_count"),
        "receipt": config.get("receipt"),
        "facts": config.get("facts"),
        "policy": config.get("policy"),
    }
    _validate_body(body)
    payload = (
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    digest = _sha256_bytes(payload)
    status = "VERIFIED_EXISTING"
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ResearchLedgerError("Research ledger target is not a regular file.")
        if target.read_bytes() != payload:
            raise ResearchLedgerError(
                "Research ledger target already exists with different approved data."
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_parents(root, target)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        status = "WRITTEN"

    return {
        "schema_version": "mmm/research-ledger-write-receipt-v1",
        "module_id": module.module_id,
        "status": status,
        "target_path": relative.as_posix(),
        "bytes": len(payload),
        "sha256": digest,
        "fact_count": len(body["facts"]),
        "shard_sha256": body["receipt"]["shard_sha256"],
        "corpus_sha256": body["receipt"]["facts_sha256"],
        "generated_java_or_gameplay_feature": False,
    }


def select_module_research_context(
    research_modules: Iterable[ProductionModule],
    *,
    query: str,
    byte_budget: int = _DEFAULT_CONTEXT_BYTES,
) -> dict[str, Any]:
    """Retrieve relevant safe facts from the complete approved shard ledger.

    Every fact remains addressable in the ledger.  A single model call receives a
    relevance-ranked byte-bounded page plus corpus/fact-count receipts; this is a
    per-call RAG budget, not a project-size or corpus-size cap.
    """

    if type(byte_budget) is not int or byte_budget < 1024:
        raise ValueError("Research context byte_budget must be at least 1024.")
    modules = sorted(
        (module for module in research_modules if is_research_shard(module)),
        key=lambda module: (
            int(module.config.get("shard_index", 0)),
            module.module_id,
        ),
    )
    facts: list[dict[str, Any]] = []
    corpus_hashes: set[str] = set()
    expected_total = 0
    declared_shard_counts: set[int] = set()
    seen_shard_indices: list[int] = []
    shard_receipts: list[dict[str, Any]] = []
    for module in modules:
        config = module.config
        receipt = config.get("receipt")
        page = config.get("facts")
        if not isinstance(receipt, dict) or not isinstance(page, list):
            raise ResearchLedgerError("Research shard receipt or facts are invalid.")
        corpus = receipt.get("facts_sha256")
        if not isinstance(corpus, str) or not corpus.startswith("sha256:"):
            raise ResearchLedgerError("Research shard corpus receipt is invalid.")
        corpus_hashes.add(corpus)
        shard_count = config.get("shard_count")
        shard_index = config.get("shard_index")
        if type(shard_count) is not int or shard_count < 1:
            raise ResearchLedgerError("Research shard_count is invalid.")
        if type(shard_index) is not int or not 0 <= shard_index < shard_count:
            raise ResearchLedgerError("Research shard_index is invalid.")
        declared_shard_counts.add(shard_count)
        seen_shard_indices.append(shard_index)
        declared_total = receipt.get("fact_count")
        if type(declared_total) is not int or declared_total < 0:
            raise ResearchLedgerError("Research shard fact_count is invalid.")
        expected_total = max(expected_total, declared_total)
        shard_hash = receipt.get("shard_sha256")
        if shard_hash != _sha256_json(page):
            raise ResearchLedgerError("Research shard facts do not match their receipt.")
        artifact = config.get("artifact")
        target_path = artifact.get("target_path") if isinstance(artifact, dict) else None
        shard_receipts.append(
            {
                "module_id": module.module_id,
                "shard_index": config.get("shard_index"),
                "shard_sha256": shard_hash,
                "target_path": target_path,
            }
        )
        facts.extend(item for item in page if isinstance(item, dict))
    if len(corpus_hashes) > 1:
        raise ResearchLedgerError("Research modules contain multiple corpus hashes.")
    if modules:
        if declared_shard_counts != {len(modules)}:
            raise ResearchLedgerError("Research ledger shard_count is incomplete.")
        if seen_shard_indices != list(range(len(modules))):
            raise ResearchLedgerError("Research ledger shard indices are incomplete.")

    fact_by_id: dict[str, dict[str, Any]] = {}
    for fact in facts:
        fact_id = fact.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id:
            raise ResearchLedgerError("Research fact_id is invalid.")
        if fact_id in fact_by_id and fact_by_id[fact_id] != fact:
            raise ResearchLedgerError("Research ledger contains a conflicting fact_id.")
        fact_by_id[fact_id] = fact
    ordered_facts = list(fact_by_id.values())
    if expected_total and len(ordered_facts) != expected_total:
        raise ResearchLedgerError(
            "Research ledger is incomplete: approved fact_count does not match pages."
        )
    corpus_sha256 = next(iter(corpus_hashes), "")
    if ordered_facts and _sha256_json(ordered_facts) != corpus_sha256:
        raise ResearchLedgerError(
            "Research ledger facts do not match the approved corpus receipt."
        )

    query_tokens = _tokens(query)
    records = _context_records(ordered_facts, byte_budget=byte_budget)
    ranked = sorted(
        records,
        key=lambda record: (
            -_record_score(record, query_tokens),
            str(record.get("record_id", "")),
        ),
    )
    base = {
        "schema_version": _CONTEXT_SCHEMA,
        "selection": "deterministic_atomic_record_relevance_v2",
        "corpus_sha256": corpus_sha256,
        "ledger_fact_count": len(ordered_facts),
        "ledger_record_count": len(records),
        "shard_count": len(modules),
        "selected_record_count": 0,
        "selected_fact_count": 0,
        "selected_facts_sha256": _sha256_json([]),
        "records": [],
        "policy": {
            "facts_are_data_not_instructions": True,
            "selected_records_are_atomic": True,
            "unselected_facts_remain_in_approved_ledger": True,
        },
    }
    selected: list[dict[str, Any]] = []
    for record in ranked:
        candidate = [*selected, record]
        candidate_fact_ids = _selected_fact_ids(candidate)
        payload = {
            **base,
            "selected_record_count": len(candidate),
            "selected_fact_count": len(candidate_fact_ids),
            "selected_facts_sha256": _sha256_json(candidate_fact_ids),
            "records": candidate,
        }
        if _json_size(payload) > byte_budget:
            continue
        selected = candidate
    selected_fact_ids = _selected_fact_ids(selected)
    result = {
        **base,
        "selected_record_count": len(selected),
        "selected_fact_count": len(selected_fact_ids),
        "selected_facts_sha256": _sha256_json(selected_fact_ids),
        "records": selected,
        "omitted_fact_count": len(ordered_facts) - len(selected_fact_ids),
        "ledger_shards": shard_receipts,
    }
    # Keep the model-facing page bounded even when the ledger has many shard
    # receipts.  The full list is code-owned and already present in module deps;
    # one aggregate receipt is sufficient in the request.
    if _json_size(result) > byte_budget:
        result.pop("ledger_shards")
        result["ledger_shards_sha256"] = _sha256_json(shard_receipts)
    while selected and _json_size(result) > byte_budget:
        selected.pop()
        selected_fact_ids = _selected_fact_ids(selected)
        result.update(
            {
                "selected_record_count": len(selected),
                "selected_fact_count": len(selected_fact_ids),
                "selected_facts_sha256": _sha256_json(selected_fact_ids),
                "records": list(selected),
                "omitted_fact_count": len(ordered_facts)
                - len(selected_fact_ids),
            }
        )
    if _json_size(result) > byte_budget:
        raise ResearchLedgerError("Research context receipt exceeds its byte budget.")
    return result


def _safe_target_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ResearchLedgerError("Research ledger target_path is invalid.")
    normalized = value.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ResearchLedgerError("Research ledger target_path escapes its root.")
    try:
        candidate.relative_to(_ALLOWED_ROOT)
    except ValueError as exc:
        raise ResearchLedgerError(
            "Research ledger target_path must stay under .minecraft_ai/research."
        ) from exc
    if candidate.suffix != ".json":
        raise ResearchLedgerError("Research ledger target must be JSON.")
    return candidate


def _reject_symlink_parents(root: Path, target: Path) -> None:
    current = target.parent
    while current != root:
        if current.exists() and current.is_symlink():
            raise ResearchLedgerError("Research ledger parent may not be a symlink.")
        current = current.parent


def _validate_body(body: dict[str, Any]) -> None:
    if type(body.get("shard_index")) is not int or body["shard_index"] < 0:
        raise ResearchLedgerError("Research shard_index is invalid.")
    if type(body.get("shard_count")) is not int or body["shard_count"] < 1:
        raise ResearchLedgerError("Research shard_count is invalid.")
    if body["shard_index"] >= body["shard_count"]:
        raise ResearchLedgerError("Research shard_index exceeds shard_count.")
    facts = body.get("facts")
    receipt = body.get("receipt")
    if not isinstance(facts, list) or not isinstance(receipt, dict):
        raise ResearchLedgerError("Research shard facts or receipt are invalid.")
    if receipt.get("shard_fact_count") != len(facts):
        raise ResearchLedgerError("Research shard_fact_count does not match facts.")
    if receipt.get("shard_sha256") != _sha256_json(facts):
        raise ResearchLedgerError("Research shard_sha256 does not match facts.")


def _tokens(value: Any) -> set[str]:
    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return {
        token.lower()
        for pattern in (_TOKEN, _SUBTOKEN)
        for token in pattern.findall(rendered)
    }


def _fact_score(fact: dict[str, Any], query_tokens: set[str]) -> int:
    # Values carry the evidence meaning; JSON field names such as
    # ``depends_on`` are common across thousands of facts and must not outrank a
    # late fact whose value directly matches the requested feature.
    value_overlap = query_tokens & _tokens(fact.get("value"))
    metadata_overlap = query_tokens & _tokens(
        {
            "source_type": fact.get("source_type"),
            "path": fact.get("path"),
        }
    )
    score = len(value_overlap) * 1_000 + len(metadata_overlap) * 10
    source_type = str(fact.get("source_type", ""))
    if source_type in {
        "research_brief_manifest",
        "technical_manifest",
        "ecosystem_manifest",
        "technology_policy",
    }:
        score += 10
    return score


def _context_records(
    facts: list[dict[str, Any]],
    *,
    byte_budget: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        source_id = str(fact.get("source_id", ""))
        by_source.setdefault(source_id, []).append(fact)

    records: list[dict[str, Any]] = []
    for source_id, source_facts in by_source.items():
        source_type = str(source_facts[0].get("source_type", ""))
        groups = (
            _technical_context_groups(source_facts)
            if source_type == "technical_query"
            else [("source", source_facts)]
        )
        for group_name, group_facts in groups:
            if group_facts:
                records.extend(
                    _bounded_context_records(
                        source_id=source_id,
                        source_type=source_type,
                        group_name=group_name,
                        facts=group_facts,
                        byte_budget=byte_budget,
                    )
                )
    return records


def _bounded_context_records(
    *,
    source_id: str,
    source_type: str,
    group_name: str,
    facts: list[dict[str, Any]],
    byte_budget: int,
) -> list[dict[str, Any]]:
    full = _context_record(
        source_id=source_id,
        source_type=source_type,
        group_name=group_name,
        facts=facts,
    )
    # Leave room for the corpus receipt and selection policy around one record.
    record_budget = max(1024, byte_budget - 1800)
    if _json_size(full) <= record_budget:
        return [full]

    anchors = _bounded_record_anchors(
        source_id=source_id,
        source_type=source_type,
        group_name=group_name,
        facts=facts,
        record_budget=record_budget,
        source_record_sha256=full["record_sha256"],
    )
    anchor_ids = {str(fact.get("fact_id", "")) for fact in anchors}
    remaining = [
        fact
        for fact in facts
        if str(fact.get("fact_id", "")) not in anchor_ids
    ]
    if not remaining:
        remaining = list(facts)
        anchors = []

    pages: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for fact in remaining:
        candidate = [*anchors, *current, fact]
        probe = _context_record(
            source_id=source_id,
            source_type=source_type,
            group_name=f"{group_name}:fragment",
            facts=candidate,
            complete=False,
            source_record_sha256=full["record_sha256"],
        )
        if _json_size(probe) <= record_budget:
            current.append(fact)
            continue
        if not current:
            raise ResearchLedgerError(
                "One research record fragment exceeds its model-page budget."
            )
        pages.append([*anchors, *current])
        current = [fact]
    if current:
        pages.append([*anchors, *current])

    return [
        _context_record(
            source_id=source_id,
            source_type=source_type,
            group_name=f"{group_name}:fragment:{index + 1}:{len(pages)}",
            facts=page,
            complete=False,
            source_record_sha256=full["record_sha256"],
            fragment_index=index,
            fragment_count=len(pages),
        )
        for index, page in enumerate(pages)
    ]


def _bounded_record_anchors(
    *,
    source_id: str,
    source_type: str,
    group_name: str,
    facts: list[dict[str, Any]],
    record_budget: int,
    source_record_sha256: str,
) -> list[dict[str, Any]]:
    """Choose repeatable metadata without letting anchors consume a page.

    Candidate safety metadata is kept whole and repeated when it fits.  A large
    value (for example license evidence) stays in ``remaining`` and is therefore
    losslessly split into ordinary value-part fragments instead of being copied
    into every fragment.
    """

    facts_by_path: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        facts_by_path.setdefault(str(fact.get("path", "")), []).append(fact)
    candidate_paths = [
        path
        for path in _ordered_anchor_paths(source_type, tuple(facts_by_path))
        if path in facts_by_path
    ]
    # Preserve at least half of the record for the fragment payload.  This also
    # bounds repeated metadata independently of the number or size of fields.
    anchor_budget = max(384, record_budget // 2)
    anchors: list[dict[str, Any]] = []
    for path in candidate_paths:
        proposed = [*anchors, *facts_by_path[path]]
        probe = _context_record(
            source_id=source_id,
            source_type=source_type,
            group_name=f"{group_name}:anchors",
            facts=proposed,
            complete=True,
            source_record_sha256=source_record_sha256,
        )
        if _json_size(probe) <= anchor_budget:
            anchors = proposed
    return anchors


def _ordered_anchor_paths(
    source_type: str,
    available_paths: tuple[str, ...],
) -> tuple[str, ...]:
    if source_type == "ecosystem_candidate":
        # Exact paths are intentional: free-form license evidence and URLs are
        # useful evidence payloads, but are not mandatory safety anchors and may
        # be arbitrarily large.
        return (
            "/candidate_id",
            "/provider",
            "/resource_kind",
            "/license/id",
            "/license/policy",
            "/minecraft_version",
            "/loader",
            "/compatibility",
            "/reuse_status",
            "/evidence_sha256",
            "/revision_sha",
            "/access/private",
            "/access/gated",
            "/access/disabled",
            "/format/has_safetensors",
            "/format/has_gguf",
            "/format/has_onnx",
            "/format/unsafe_serialization_file_count",
            "/format/repository_code_file_count",
        )

    prefixes = _record_anchor_prefixes(source_type)
    return tuple(
        path
        for prefix in prefixes
        for path in available_paths
        if path == prefix or path.startswith(prefix + "/")
    )


def _record_anchor_prefixes(source_type: str) -> tuple[str, ...]:
    if source_type == "technology_requirement":
        return (
            "/requirement_id",
            "/domain_id",
            "/capability_kind",
            "/target",
            "/authority",
            "/hardware",
            "/latency",
            "/privacy",
            "/offline_required",
            "/required_gates",
            "/required_tests",
            "/deterministic_fallback",
        )
    return (
        "/domain_id",
        "/document_id",
        "/candidate_id",
        "/requirement_id",
        "/query_sha256",
    )


def _technical_context_groups(
    facts: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    summary: list[dict[str, Any]] = []
    hits: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        path = str(fact.get("path", ""))
        match = _TECHNICAL_HIT_PATH.match(path)
        if match is None:
            summary.append(fact)
        else:
            hits.setdefault(match.group(1), []).append(fact)
    if not hits:
        return [("query", summary)]

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    if summary:
        groups.append(("query", summary))
    for prefix, hit_facts in sorted(hits.items()):
        if prefix.startswith("primary/"):
            branch = "/primary/"
        else:
            branch = "/" + "/".join(prefix.split("/")[:2]) + "/"
        supporting = [
            fact
            for fact in summary
            if (
                not str(fact.get("path", "")).startswith(
                    ("/primary/", "/corrections/")
                )
                or str(fact.get("path", "")).startswith(branch)
            )
        ]
        groups.append((prefix.replace("/", ":"), [*supporting, *hit_facts]))
    return groups


def _context_record(
    *,
    source_id: str,
    source_type: str,
    group_name: str,
    facts: list[dict[str, Any]],
    complete: bool = True,
    source_record_sha256: str | None = None,
    fragment_index: int | None = None,
    fragment_count: int | None = None,
) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for fact in facts:
        fact_id = str(fact.get("fact_id", ""))
        unique.setdefault(fact_id, fact)
    ordered = list(unique.values())
    fields: dict[str, Any] = {}
    by_path: dict[str, list[dict[str, Any]]] = {}
    for fact in ordered:
        by_path.setdefault(str(fact.get("path", "")), []).append(fact)
    for path, parts in sorted(by_path.items()):
        parts.sort(key=lambda item: int(item.get("value_part_index", 0)))
        declared = {int(item.get("value_part_count", 0)) for item in parts}
        indices = [int(item.get("value_part_index", -1)) for item in parts]
        declared_count = next(iter(declared), 0)
        parts_complete = (
            len(declared) == 1
            and declared_count == len(parts)
            and indices == list(range(len(parts)))
        )
        if complete and not parts_complete:
            raise ResearchLedgerError(
                f"Research record has incomplete value parts: {source_id}{path}"
            )
        value_type = str(parts[0].get("value_type", ""))
        if any(str(item.get("value_type", "")) != value_type for item in parts):
            raise ResearchLedgerError("Research record value types conflict.")
        value = (
            "".join(str(item.get("value", "")) for item in parts)
            if value_type == "string"
            else parts[0].get("value")
        )
        fields[path] = (
            value
            if parts_complete
            else {
                "value_fragment": value,
                "value_type": value_type,
                "part_indices": indices,
                "part_count": declared_count,
            }
        )
    fact_ids = [str(fact.get("fact_id", "")) for fact in ordered]
    record_seed = {
        "source_id": source_id,
        "group": group_name,
        "fact_ids": fact_ids,
    }
    record = {
        "record_id": "research_record:"
        + _sha256_json(record_seed).removeprefix("sha256:")[:24],
        "source_id": source_id,
        "source_type": source_type,
        "group": group_name,
        "complete": complete,
        "fact_count": len(fact_ids),
        "fact_ids": fact_ids,
        "record_sha256": _sha256_json(ordered),
        "fields": fields,
    }
    if source_record_sha256 is not None:
        record["source_record_sha256"] = source_record_sha256
    if fragment_index is not None and fragment_count is not None:
        record["fragment_index"] = fragment_index
        record["fragment_count"] = fragment_count
    return record


def _record_score(record: dict[str, Any], query_tokens: set[str]) -> int:
    fields = record.get("fields")
    fields = fields if isinstance(fields, dict) else {}
    value_tokens = _tokens(list(fields.values()))
    path_tokens = _tokens(list(fields))
    score = len(query_tokens & value_tokens) * 1_000
    score += len(query_tokens & path_tokens) * 10
    if record.get("source_type") in {
        "research_brief_manifest",
        "technical_manifest",
        "ecosystem_manifest",
        "technology_policy",
    }:
        score += 10
    return score


def _selected_fact_ids(records: list[dict[str, Any]]) -> list[str]:
    values = {
        str(fact_id)
        for record in records
        for fact_id in record.get("fact_ids", [])
        if isinstance(fact_id, str)
    }
    return sorted(values)


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()
