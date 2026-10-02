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


def _public_contract(source: str, *, include_package: bool = False) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key not in {"declaration", "initialized"}}
        for row in class_body_member_contracts(source)
        if row.get("visibility") in {"public", "protected"}
        or (include_package and row.get("visibility") != "private")
    ]


def _private_implementation(source: str) -> bool:
    rows = class_body_member_contracts(source)
    return bool(rows) and all(
        row.get("visibility") == "private" and row.get("kind") in {"field", "method"}
        for row in rows
    )


def _private_nested_type(source: str) -> bool:
    rows = class_body_member_contracts(source)
    return bool(rows) and all(
        row.get("visibility") == "private" and row.get("kind") == "type"
        for row in rows
    )


@dataclass(frozen=True)
class RegionCorrection:
    """Protect accepted declarations without freezing a rejected private design."""

    chunks: tuple[str, ...]
    selected: frozenset[int]
    allow_private_restructure: bool = False
    allow_private_type_additions: bool = False

    @classmethod
    def for_diagnostic(
        cls,
        source: str,
        diagnostic: str,
        *,
        allow_private_restructure: bool = False,
        allow_private_type_additions: bool = False,
    ) -> RegionCorrection | None:
        from .atomic_concern_source import _structure_scan

        chunks = strict_member_chunks(source)
        if not chunks or any(not _identity(chunk) for chunk in chunks):
            # Initializer blocks lack declaration identities; use the enclosing
            # concern correction boundary rather than guessing a source offset.
            return None
        match = re.search(r"reassigns final field\(s\):\s*(.+)$", diagnostic)
        if match and "\n" not in diagnostic and not allow_private_restructure:
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
        return (
            cls(
                chunks,
                selected,
                allow_private_restructure,
                allow_private_type_additions,
            )
            if selected
            else None
        )

    def payload(self) -> dict:
        return {
            "selected_declarations": [self.chunks[i] for i in sorted(self.selected)],
            "immutable_declarations": [chunk for i, chunk in enumerate(self.chunks) if i not in self.selected],
            "allow_private_restructure": self.allow_private_restructure,
            "allow_private_type_additions": self.allow_private_type_additions,
            "rules": (
                "Return corrected selected_declarations only, as complete Java class-body members. "
                "Do not return immutable_declarations. "
                + (
                    "This current concern candidate has NEVER been accepted. Its private fields/methods "
                    "are provisional implementation choices, not required APIs. Rebuild those private "
                    "members from task_authority when necessary; update all local callers together. "
                    "You may remove invented, unused helpers unrelated to the selected requirements "
                    "and add private helpers needed to implement those requirements. Preserve all "
                    "non-private contracts and existing nested types. "
                    + (
                        "For this explicit type-authority correction only, you may add a private "
                        "nested class/record whose exact name appears in the host diagnostic when "
                        "that authored concern owns the runtime domain object. "
                        if self.allow_private_type_additions else
                        "Do not add nested types. "
                    )
                    + "Do not implement a sibling concern just to retain an invented helper. "
                    "Never fabricate a dependency API. "
                    if self.allow_private_restructure else
                    "The host merges members into their original slots. Keep every declaration "
                    "identity and public/protected API. "
                )
                + "Preserve authored behavior, "
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
        additions: list[str] = []
        seen_symbols: set[str] = set()
        frozen_symbols = {
            symbol for i, chunk in enumerate(self.chunks)
            if i not in self.selected or not _private_implementation(chunk)
            for symbol in _identity(chunk)
        }
        candidate_chunks = strict_member_chunks(candidate)
        for chunk in candidate_chunks:
            identity = _identity(chunk)
            if not identity or seen_symbols.intersection(identity):
                raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: duplicate or unidentified declaration")
            seen_symbols.update(identity)
            index = original.get(identity)
            if index is None:
                allowed_private_addition = (
                    self.allow_private_restructure
                    and (
                        _private_implementation(chunk)
                        or (
                            self.allow_private_type_additions
                            and _private_nested_type(chunk)
                        )
                    )
                    and not frozen_symbols.intersection(identity)
                )
                if allowed_private_addition:
                    additions.append(chunk)
                    continue
                raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: new declaration outside private implementation")
            if index not in self.selected:
                if chunk != self.chunks[index]:
                    raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: immutable declaration changed")
                continue
            if _public_contract(chunk, include_package=self.allow_private_restructure) != _public_contract(
                self.chunks[index], include_package=self.allow_private_restructure,
            ):
                raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: public declaration contract changed")
            if (
                self.allow_private_restructure
                and any(row.get("kind") == "type" for row in class_body_member_contracts(self.chunks[index]))
                and chunk != self.chunks[index]
            ):
                raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: nested type changed during private reconstruction")
            if (
                self.allow_private_restructure
                and _private_implementation(self.chunks[index])
                and not _private_implementation(chunk)
            ):
                raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: private implementation visibility changed")
            replacements[index] = chunk
        required = {
            i for i in self.selected
            if not self.allow_private_restructure or not _private_implementation(self.chunks[i])
        }
        if not required.issubset(replacements):
            raise CustomModuleGenerationError("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: selected declaration missing")
        if self.allow_private_restructure and self.selected == frozenset(range(len(self.chunks))):
            # A reconstructed private field initializer may depend on a new field.
            # Preserve the supplied declaration order instead of appending that
            # new field after its first use in an initializer.
            return "\n\n".join(candidate_chunks)
        return "\n\n".join([
            replacements.get(i, chunk) for i, chunk in enumerate(self.chunks)
            if i not in self.selected or i in replacements
        ] + additions)
