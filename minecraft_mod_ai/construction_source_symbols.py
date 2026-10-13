"""Index hash-checked installed Java source into callable, owner-qualified symbols.

Only unambiguous imports and real declarations are usable. This does not
authorize reference-only/uninstalled code or bypass target compilation.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from .java_operation_ir import PRIMITIVES
from .construction_context import java_source_units

_GENERIC = re.compile(r"<[^<>]*>")
_QUALIFIED = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*")


def _raw_type(value):
    value = str(value).strip().replace("...", "[]")
    while "<" in value or ">" in value:
        reduced = _GENERIC.sub("", value)
        if reduced == value:
            return ""
        value = reduced
    return re.sub(r"\s+", "", value)


def _resolve(value, unit, known):
    raw = _raw_type(value)
    array = ""
    while raw.endswith("[]"):
        raw, array = raw[:-2], array + "[]"
    if raw in PRIMITIVES:
        return raw + array
    if not _QUALIFIED.fullmatch(raw):
        return ""
    imports = [x for x in unit.get("imports", ()) if not x.startswith("static ")]
    explicit = {x.rsplit(".", 1)[-1]: x for x in imports if not x.endswith(".*")}
    candidates = []
    if raw in known:
        candidates.append(raw)
    if raw in explicit and explicit[raw] in known:
        candidates.append(explicit[raw])
    package = unit["owner"].rpartition(".")[0]
    if (package + "." + raw) in known:
        candidates.append(package + "." + raw)
    if ("java.lang." + raw) in known:
        candidates.append("java.lang." + raw)
    # Wildcard imports are not granted arbitrary symbol authority: only unique
    # actually inspected matching owners qualify.
    for item in imports:
        if item.endswith(".*") and (item[:-1] + raw) in known:
            candidates.append(item[:-1] + raw)
    unique = set(candidates)
    return next(iter(unique)) + array if len(unique) == 1 else ""


def inspected_project_source_symbols(source_pages, inspected_symbols):
    if not isinstance(source_pages, Mapping):
        raise ValueError("CONSTRUCTION_SOURCE_PAGES_INVALID")
    symbols = {}
    rejected = []
    units = [unit for collection in source_pages.values() for unit in collection]
    known = set(inspected_symbols)
    for unit in units:
        if unit["kind"] in {"class", "interface", "enum", "record"}:
            known.add(unit["owner"])
    for unit in units:
        owner, kind = unit["owner"], unit["kind"]
        if kind not in {"class", "interface", "enum", "record"}:
            continue
        if owner in inspected_symbols:
            raise ValueError("CONSTRUCTION_DUPLICATE_PROJECT_CLASS: " + owner)
        symbols[owner] = {
            "kind": "CLASS", "name": owner.rsplit(".", 1)[-1], "owner": owner,
            "access": (1 if unit["public"] else 0) |
                      (0x200 if kind == "interface" else 0) |
                      (0x400 if kind == "interface" or
                       re.search(r"\babstract\b", unit["signature"]) else 0),
            "superclass": ("java.lang.Object"
                           if kind != "interface" and owner != "java.lang.Object"
                           else ""),
            "interfaces": [], "side": unit["side"], "source_hash": unit["source_sha256"],
            "source_path": unit["path"], "provenance": "materialized_source",
        }
    all_known = known | set(symbols)
    for unit in units:
        kind = unit["kind"]
        if kind not in {"method", "constructor", "field"} or not unit["public"]:
            continue
        owner = unit["owner"]
        if owner not in symbols:
            continue
        member = unit["name"]
        if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", member):
            rejected.append({"path": unit["path"], "name": member, "reason": "ambiguous_name"})
            continue
        parameters = []
        for param in unit["parameters"]:
            type_ = _resolve(param["type"], unit, all_known)
            if not type_:
                rejected.append({"path": unit["path"], "name": member,
                                 "reason": "unresolved_parameter_type", "type": param["type"]})
                break
            parameters.append({"name": param["name"], "type": type_})
        else:
            return_type = (owner if kind == "constructor" else
                           _resolve(unit["return_type"], unit, all_known))
            if not return_type:
                rejected.append({"path": unit["path"], "name": member,
                                 "reason": "unresolved_return_type", "type": unit["return_type"]})
                continue
            public = bool(re.search(r"\bpublic\b", unit["signature"]))
            access = 1 if public else 4
            symbol_id = owner + "#" + member + "(" + ",".join(x["type"] for x in parameters) + ")"
            if symbol_id in symbols:
                raise ValueError("CONSTRUCTION_DUPLICATE_SOURCE_SIGNATURE: " + symbol_id)
            symbols[symbol_id] = {
                "kind": kind.upper(), "owner": owner, "name": member,
                "parameters": parameters, "return_type": return_type,
                "access": access, "static": unit["static"], "side": unit["side"],
                "source_path": unit["path"], "source_hash": unit["source_sha256"],
                "provenance": "materialized_source",
            }
    return symbols, tuple(rejected)
