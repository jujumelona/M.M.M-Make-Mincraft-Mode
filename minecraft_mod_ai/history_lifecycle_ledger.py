from __future__ import annotations

"""Semantic lifecycle index for plans, executions, verification and decisions.

Existing stores remain authoritative for proposal bodies, planning snapshots,
transcripts, trajectories, verifier receipts and root-cause payloads. This module
stores only stable references and relations, preventing a second copy of those
payloads while making past attempts and their current validity queryable.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mmm/history-lifecycle-ledger-v1"


class HistoryKind(str, Enum):
    PLAN = "plan"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    FAILURE = "failure"
    DECISION = "decision"
    UNRESOLVED = "unresolved"


class HistoryStatus(str, Enum):
    ACTIVE = "active"
    OBSERVED = "observed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    RESOLVED = "resolved"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


_KINDS = {item.value for item in HistoryKind}
_STATUSES = {item.value for item in HistoryStatus}
_TERMINAL = {"rejected", "resolved", "superseded", "invalidated"}
_VERIFICATION_LEVELS = {f"L{index}" for index in range(6)}
_ALLOWED_RELATIONS = {
    "derived_from",
    "supersedes",
    "verified_by",
    "invalidates",
    "decided_by",
    "resolves",
    "related_to",
}
_TERMINATING_RELATIONS = {"supersedes", "invalidates", "resolves"}


@dataclass(frozen=True)
class ArtifactRef:
    """Pointer into an existing authoritative store; payloads never live here."""

    store: str
    ref: str
    digest: str = ""

    def validate(self) -> None:
        if not self.store.strip() or not self.ref.strip():
            raise ValueError("HISTORY_ARTIFACT_REF")
        if self.digest and len(self.digest) < 16:
            raise ValueError("HISTORY_ARTIFACT_DIGEST")


@dataclass(frozen=True)
class HistoryRelation:
    relation: str
    record_id: str

    def validate(self) -> None:
        if self.relation not in _ALLOWED_RELATIONS:
            raise ValueError(f"HISTORY_RELATION:{self.relation}")
        if not self.record_id.strip():
            raise ValueError("HISTORY_RELATION_RECORD_ID")


@dataclass(frozen=True)
class HistoryRecord:
    record_id: str
    subject_id: str
    kind: str
    status: str
    summary: str
    artifact_refs: tuple[ArtifactRef, ...] = ()
    relations: tuple[HistoryRelation, ...] = ()
    verification_level: str = "L0"
    failure_taxonomy: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    retest_triggers: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        if not self.record_id.strip() or not self.subject_id.strip():
            raise ValueError("HISTORY_ID")
        if self.kind not in _KINDS:
            raise ValueError(f"HISTORY_KIND:{self.kind}")
        if self.status not in _STATUSES:
            raise ValueError(f"HISTORY_STATUS:{self.status}")
        if not self.summary.strip():
            raise ValueError("HISTORY_SUMMARY")
        if self.verification_level not in _VERIFICATION_LEVELS:
            raise ValueError(f"HISTORY_VERIFICATION_LEVEL:{self.verification_level}")
        if self.kind == HistoryKind.VERIFICATION.value and not self.artifact_refs:
            raise ValueError("HISTORY_VERIFICATION_EVIDENCE_REQUIRED")
        if self.status == HistoryStatus.REJECTED.value and not self.rejection_reasons:
            raise ValueError("HISTORY_REJECTION_REASON_REQUIRED")
        if self.status == HistoryStatus.INVALIDATED.value and not any(
            relation.relation == "invalidates" for relation in self.relations
        ):
            raise ValueError("HISTORY_INVALIDATION_RELATION_REQUIRED")
        for ref in self.artifact_refs:
            ref.validate()
        for relation in self.relations:
            relation.validate()


def _payload(record: HistoryRecord, *, include_record_id: bool) -> dict[str, Any]:
    value = asdict(record)
    if not include_record_id:
        value.pop("record_id", None)
    value["schema_version"] = SCHEMA_VERSION
    return value


def _stable_id(record: HistoryRecord) -> str:
    raw = json.dumps(
        _payload(record, include_record_id=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "hist:" + hashlib.sha256(raw).hexdigest()


def build_history_record(
    *,
    subject_id: str,
    kind: str,
    status: str,
    summary: str,
    artifact_refs: Sequence[ArtifactRef] = (),
    relations: Sequence[HistoryRelation] = (),
    verification_level: str = "L0",
    failure_taxonomy: Sequence[str] = (),
    rejection_reasons: Sequence[str] = (),
    retest_triggers: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> HistoryRecord:
    values = dict(
        record_id="pending",
        subject_id=subject_id,
        kind=kind,
        status=status,
        summary=summary,
        artifact_refs=tuple(artifact_refs),
        relations=tuple(relations),
        verification_level=verification_level,
        failure_taxonomy=tuple(str(value) for value in failure_taxonomy),
        rejection_reasons=tuple(str(value) for value in rejection_reasons),
        retest_triggers=tuple(str(value) for value in retest_triggers),
        metadata=tuple(sorted((str(key), str(value)) for key, value in (metadata or {}).items())),
    )
    candidate = HistoryRecord(**values)
    values["record_id"] = _stable_id(candidate)
    record = HistoryRecord(**values)
    record.validate()
    return record


def history_record_from_verification(
    *,
    subject_id: str,
    summary: str,
    verification: Mapping[str, Any],
    evidence_ref: ArtifactRef,
    relations: Sequence[HistoryRelation] = (),
    metadata: Mapping[str, Any] | None = None,
) -> HistoryRecord:
    """Index trajectory_verification output without inventing a second classifier."""

    failed = tuple(sorted({
        str(item.get("kind"))
        for item in verification.get("verifier_chain", ())
        if isinstance(item, Mapping) and str(item.get("status")) == "FAIL"
    }))
    return build_history_record(
        subject_id=subject_id,
        kind=HistoryKind.VERIFICATION.value,
        status=HistoryStatus.VERIFIED.value,
        summary=summary,
        artifact_refs=(evidence_ref,),
        relations=relations,
        verification_level=str(verification.get("level", "L0")),
        failure_taxonomy=failed if verification.get("verified_failure") is True else (),
        metadata=metadata,
    )


def _decode(payload: Mapping[str, Any]) -> HistoryRecord:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("HISTORY_SCHEMA_VERSION")
    record = HistoryRecord(
        record_id=str(payload.get("record_id", "")),
        subject_id=str(payload.get("subject_id", "")),
        kind=str(payload.get("kind", "")),
        status=str(payload.get("status", "")),
        summary=str(payload.get("summary", "")),
        artifact_refs=tuple(ArtifactRef(**item) for item in payload.get("artifact_refs", ())),
        relations=tuple(HistoryRelation(**item) for item in payload.get("relations", ())),
        verification_level=str(payload.get("verification_level", "L0")),
        failure_taxonomy=tuple(str(value) for value in payload.get("failure_taxonomy", ())),
        rejection_reasons=tuple(str(value) for value in payload.get("rejection_reasons", ())),
        retest_triggers=tuple(str(value) for value in payload.get("retest_triggers", ())),
        metadata=tuple((str(key), str(value)) for key, value in payload.get("metadata", ())),
    )
    record.validate()
    if _stable_id(record) != record.record_id:
        raise ValueError("HISTORY_RECORD_HASH_MISMATCH")
    return record


class HistoryLifecycleLedger:
    """Append-only relation index with deterministic current-state projection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read_all(self) -> list[HistoryRecord]:
        if not self.path.exists():
            return []
        rows: list[HistoryRecord] = []
        seen: set[str] = set()
        for line_number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                record = _decode(json.loads(raw))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"HISTORY_CORRUPT_LINE:{line_number}") from exc
            if record.record_id in seen:
                raise ValueError(f"HISTORY_DUPLICATE_RECORD:{record.record_id}")
            seen.add(record.record_id)
            rows.append(record)
        return rows

    def append(self, record: HistoryRecord) -> bool:
        record.validate()
        rows = self.read_all()
        existing = {item.record_id for item in rows}
        if record.record_id in existing:
            return False
        for relation in record.relations:
            if relation.record_id not in existing:
                raise ValueError(f"HISTORY_UNKNOWN_RELATION:{relation.record_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            _payload(record, include_record_id=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return True

    def current(self, *, subject_id: str | None = None) -> list[HistoryRecord]:
        rows = self.read_all()
        terminated = {
            relation.record_id
            for row in rows
            for relation in row.relations
            if relation.relation in _TERMINATING_RELATIONS
        }
        return [
            row
            for row in rows
            if row.record_id not in terminated
            and row.status not in _TERMINAL
            and (subject_id is None or row.subject_id == subject_id)
        ]

    def subject_history(self, subject_id: str) -> list[HistoryRecord]:
        return [row for row in self.read_all() if row.subject_id == subject_id]

    def related(self, record_id: str) -> list[HistoryRecord]:
        return [
            row
            for row in self.read_all()
            if row.record_id == record_id
            or any(relation.record_id == record_id for relation in row.relations)
        ]

    def compact_view(self, *, subject_id: str | None = None, limit: int = 64) -> dict[str, Any]:
        """Bounded small-model view; referenced stores retain complete evidence."""

        if limit < 1:
            raise ValueError("HISTORY_VIEW_LIMIT")
        rows = self.read_all()
        if subject_id is not None:
            rows = [row for row in rows if row.subject_id == subject_id]
        current_ids = {row.record_id for row in self.current(subject_id=subject_id)}
        return {
            "schema_version": SCHEMA_VERSION,
            "record_count": len(rows),
            "records": [
                {
                    "record_id": row.record_id,
                    "subject_id": row.subject_id,
                    "kind": row.kind,
                    "status": row.status,
                    "current": row.record_id in current_ids,
                    "summary": row.summary,
                    "verification_level": row.verification_level,
                    "artifact_refs": [asdict(ref) for ref in row.artifact_refs],
                    "relations": [asdict(relation) for relation in row.relations],
                    "failure_taxonomy": list(row.failure_taxonomy),
                    "rejection_reasons": list(row.rejection_reasons),
                    "retest_triggers": list(row.retest_triggers),
                }
                for row in rows[-limit:]
            ],
        }


__all__ = [
    "ArtifactRef",
    "HistoryKind",
    "HistoryLifecycleLedger",
    "HistoryRecord",
    "HistoryRelation",
    "HistoryStatus",
    "SCHEMA_VERSION",
    "build_history_record",
    "history_record_from_verification",
]
