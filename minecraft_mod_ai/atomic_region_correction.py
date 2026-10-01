"""Keep a rejected Java candidate while correcting its diagnostic-owned members."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .custom_module_errors import CustomModuleGenerationError
from .java_region_parser import (
    class_body_assignment_targets,
    class_body_member_contracts,
    strict_member_chunks,
)


def _identity(source: str) -> tuple[str, ...]:
    from .atomic_concern_source import _member_declaration_symbols

    return tuple(sorted(_member_declaration_symbols(source)))


def _public_contract(source: str) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key not in {"declaration", "initialized"}}
        for row in class_body_member_contracts(source)
        if row.get("visibility") in {"public", "protected"}
    ]


@dataclass(frozen=True)
class RegionCorrection:
    """Original chunks are immutable; only selected declaration slots may change."""

    chunks: tuple[str, ...]
    selected: frozenset[int]

    @classmethod
    def for_diagnostic(cls, source: str, diagnostic: str) -> RegionCorrection | None:
        from .atomic_concern_source import _structure_scan

        chunks = strict_member_chunks(source)
        if not chunks or any(not _identity(chunk) for chunk in chunks):
            # Initializer blocks lack declaration identities; use the enclosing
            # concern correction boundary rather than guessing a source offset.
            return None
        match = re.search(r"reassigns final field\(s\):\s*(.+)$", diagnostic)
        if match:
            names = {name.strip() for name in match[1].split(",")}
            selected = frozenset(
                index for index, chunk in enumerate(chunks)
                if any(
                    row.get("kind") == "field" and row.get("symbol") in names
                    for row in class_body_member_contracts(chunk)
                ) or names.intersection(class_body_assignment_targets(chunk))
                or any(re.search(rf"\b{re.escape(name)}\b", _structure_scan(chunk)) for name in names)
            )
        else:
            # An unlocalized semantic diagnostic stays inside this one concern.
            # Never infer sibling ownership from a substring of an error message.
            selected = frozenset(range(len(chunks)))
        return cls(chunks, selected) if selected else None

    def payload(self) -> dict:
        return {
            "selected_declarations": [self.chunks[i] for i in sorted(self.selected)],
            "immutable_declarations": [chunk for i, chunk in enumerate(self.chunks) if i not in self.selected],
            "rules": (
                "Return corrected selected_declarations only, as complete Java class-body members. "
                "The host merges them into their original slots; do not return immutable_declarations. "
                "Keep every declaration identity and public/protected API. Preserve authored behavior, "
                "guards, thresholds and side effects. Do not silence the error by deleting required logic. "
                "For a final-field assignment, determine from the requirements whether the new private "
                "field is mutable state or the assignment is wrong; correct that inconsistency. "
                "Never change final fields owned by a sibling or dependency."
            ),
        }

    def merge(self, candidate: str) -> str:
        original = {_identity(chunk): index for index, chunk in enumerate(self.chunks)}
        if len(original) != len(self.chunks):
            raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: ambiguous declaration identities")
        replacements: dict[int, str] = {}
        for chunk in strict_member_chunks(candidate):
            index = original.get(_identity(chunk))
            if index is None or index in replacements:
                raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: new or duplicate declaration")
            if index not in self.selected:
                if chunk != self.chunks[index]:
                    raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: immutable declaration changed")
                continue
            if _public_contract(chunk) != _public_contract(self.chunks[index]):
                raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: public declaration contract changed")
            replacements[index] = chunk
        if set(replacements) != set(self.selected):
            raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: selected declaration missing")
        return "\n\n".join(replacements.get(i, chunk) for i, chunk in enumerate(self.chunks))
