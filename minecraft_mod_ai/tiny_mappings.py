"""Tiny v2 namespace and descriptor translation (Fabric Tiny v2 specification)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import re


class MappingError(ValueError):
    pass


def _unescape(value: str) -> str:
    out = []
    escapes = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "\\": "\\"}
    i = 0
    while i < len(value):
        if value[i] == "\\":
            i += 1
            if i == len(value) or value[i] not in escapes:
                raise MappingError("Invalid Tiny escape")
            out.append(escapes[value[i]])
        else:
            out.append(value[i])
        i += 1
    return "".join(out)


@dataclass
class TinyMappings:
    namespaces: tuple[str, ...]
    classes: list[tuple[str, ...]] = field(default_factory=list)
    members: list[tuple[str, str, str, tuple[str, ...]]] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> TinyMappings:
        lines = text.splitlines()
        header = lines[0].split("\t") if lines else []
        if len(header) < 5 or header[:3] != ["tiny", "2", "0"]:
            raise MappingError("Expected Tiny 2.0 with at least two namespaces")
        result = cls(tuple(header[3:]))
        if len(set(result.namespaces)) != len(result.namespaces):
            raise MappingError("Duplicate namespace")
        owner = None
        escaped = False
        seen = set()
        for line in lines[1:]:
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if owner is None and parts[0] == "":
                if parts[1] == "escaped-names":
                    escaped = True
                continue
            if parts[0] == "c":
                names = tuple(_unescape(n) if escaped else n for n in parts[1:])
                if len(names) != len(result.namespaces) or not names[0]:
                    raise MappingError("Invalid class mapping")
                key = ("class", names[0])
                if key in seen:
                    raise MappingError("Duplicate class mapping")
                seen.add(key)
                result.classes.append(names)
                owner = names[0]
            elif len(parts) > 2 and parts[:2] in (["", "m"], ["", "f"]):
                if owner is None:
                    raise MappingError("Member without owner")
                kind, descriptor = parts[1:3]
                names = tuple(_unescape(n) if escaped else n for n in parts[3:])
                if len(names) != len(result.namespaces) or not names[0]:
                    raise MappingError("Invalid member mapping")
                key = (owner, kind, names[0], descriptor)
                if key in seen:
                    raise MappingError("Duplicate member mapping")
                seen.add(key)
                result.members.append((owner, kind, descriptor, names))
            elif line.startswith(("\t\tc\t", "\tc\t", "\t\tp\t", "\t\tv\t", "\t\t\tc\t")):
                # Comments and local/parameter names do not change JVM identities.
                continue
            else:
                raise MappingError(f"Invalid Tiny record: {line!r}")
        result.classes = tuple(result.classes)
        result.members = tuple(result.members)
        return result

    def _indices(self, source: str, target: str) -> tuple[int, int]:
        try:
            return self.namespaces.index(source), self.namespaces.index(target)
        except ValueError as exc:
            raise MappingError("Unknown namespace") from exc

    @cached_property
    def _class_indexes(self):
        indexes = [{} for _ in self.namespaces]
        for row in self.classes:
            for index, name in enumerate(row):
                if not name:
                    continue
                if name in indexes[index]:
                    raise MappingError("Ambiguous class mapping")
                indexes[index][name] = row
        return indexes

    @cached_property
    def _member_indexes(self):
        indexes = [{} for _ in self.namespaces]
        for owner, kind, descriptor, names in self.members:
            for index, name in enumerate(names):
                if not name:
                    continue
                key = (owner, kind, descriptor, name)
                if key in indexes[index]:
                    raise MappingError("Ambiguous member mapping")
                indexes[index][key] = names
        return indexes

    def class_name(self, name: str, source: str, target: str) -> str:
        a, b = self._indices(source, target)
        row = self._class_indexes[a].get(name)
        if row is None:
            return name  # External library/JDK class, unchanged by these mappings.
        if not row[b]:
            raise MappingError(f"Missing {target} name for {name}")
        return row[b]

    def descriptor(self, descriptor: str, source: str, target: str) -> str:
        self._indices(source, target)
        return re.sub(r"L([^;]+);", lambda m: "L" + self.class_name(m[1], source, target) + ";", descriptor)

    def member(self, owner: str, name: str, descriptor: str, kind: str,
               source: str, target: str) -> tuple[str, str, str]:
        a, b = self._indices(source, target)
        base = self.namespaces[0]
        base_owner = self.class_name(owner, source, base)
        base_desc = self.descriptor(descriptor, source, base)
        names = self._member_indexes[a].get((base_owner, kind, base_desc, name))
        mapped = name
        if names is not None:
            mapped = names[b]
            if not mapped:
                raise MappingError("Missing target member name")
        return self.class_name(owner, source, target), mapped, self.descriptor(descriptor, source, target)
