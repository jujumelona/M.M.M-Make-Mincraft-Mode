from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

from .authored_structured_design import (
    active_concern_records,
    normalize_structured_sections,
    structured_sections_sha256,
)
from .planning_detail_template import WORKSHEET_SECTIONS

CANONICAL_CONCERN_AUTHORITY_SCHEMA = "mmm/canonical-concern-authority-v1"


class CanonicalConcernAuthority:
    """Canonical, host-owned SSOT for structured design concern records.

    Semantic concern records (such as actors, inputs, outputs, state fields)
    are finalized here BEFORE graph partitioning and Java production.
    Graph nodes distribute references/provenance only, never splitting or
    reconstructing semantic authority.
    """

    def __init__(
        self,
        sections: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
        *,
        schema_version: str = CANONICAL_CONCERN_AUTHORITY_SCHEMA,
    ) -> None:
        self.schema_version = schema_version
        cleaned: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}
        if isinstance(sections, Mapping):
            for sec_name, concerns in sections.items():
                if not isinstance(concerns, Mapping):
                    continue
                sec_dict: dict[str, tuple[dict[str, Any], ...]] = {}
                for concern_name, rows in concerns.items():
                    if isinstance(rows, Sequence) and not isinstance(
                        rows, (str, bytes, bytearray)
                    ):
                        sec_dict[str(concern_name)] = tuple(
                            deepcopy(dict(row))
                            for row in rows
                            if isinstance(row, Mapping)
                        )
                if sec_dict:
                    cleaned[str(sec_name)] = sec_dict
        self._sections: dict[str, dict[str, tuple[dict[str, Any], ...]]] = cleaned

    @classmethod
    def from_structured_sections(
        cls,
        structured_sections: Mapping[str, Any] | None,
    ) -> "CanonicalConcernAuthority":
        if not isinstance(structured_sections, Mapping):
            return cls({})

        normalized = normalize_structured_sections(structured_sections)
        sections: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}

        for section in WORKSHEET_SECTIONS:
            records_by_concern = active_concern_records(normalized, section)
            if records_by_concern:
                sections[section] = {
                    concern: tuple(records)
                    for concern, records in records_by_concern.items()
                }

        return cls(sections)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalConcernAuthority":
        if not isinstance(data, Mapping):
            return cls({})
        schema_version = str(
            data.get("schema_version") or CANONICAL_CONCERN_AUTHORITY_SCHEMA
        )
        sections = data.get("sections")
        return cls(
            sections if isinstance(sections, Mapping) else {},
            schema_version=schema_version,
        )

    def get_concern_records(
        self,
        section: str,
        concern: str,
    ) -> tuple[dict[str, Any], ...]:
        sec = self._sections.get(str(section))
        if not sec:
            return ()
        rows = sec.get(str(concern))
        return deepcopy(rows) if rows else ()

    def has_concern(self, section: str, concern: str) -> bool:
        sec = self._sections.get(str(section))
        return bool(sec and str(concern) in sec)

    def active_concerns_for_section(self, section: str) -> tuple[str, ...]:
        sec = self._sections.get(str(section))
        return tuple(sec.keys()) if sec else ()

    @property
    def sections(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return {
            sec: {
                concern: [dict(row) for row in rows]
                for concern, rows in concerns.items()
            }
            for sec, concerns in self._sections.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sections": self.sections,
        }

    def authority_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def __bool__(self) -> bool:
        return bool(self._sections)


__all__ = [
    "CANONICAL_CONCERN_AUTHORITY_SCHEMA",
    "CanonicalConcernAuthority",
]
