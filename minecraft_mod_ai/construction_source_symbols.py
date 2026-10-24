"""Index hash-checked installed Java source into callable, owner-qualified symbols.

Only unambiguous imports and real declarations are usable. This does not
authorize reference-only/uninstalled code or bypass target compilation.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from .java_operation_ir import PRIMITIVES
from .construction_type_contracts import canonical_type_name

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
    raw = canonical_type_name(_raw_type(value))
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
    prefix, separator, suffix = raw.partition('.')
    if separator and prefix in explicit and (explicit[prefix] + '.' + suffix) in known:
        candidates.append(explicit[prefix] + '.' + suffix)
    package = unit.get('package', unit["owner"].rpartition(".")[0])
    enclosing = unit['owner']
    while enclosing and enclosing != package:
        if (enclosing + '.' + raw) in known:
            candidates.append(enclosing + '.' + raw)
        enclosing = enclosing.rpartition('.')[0]
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
        modifiers = set(unit.get('modifiers', ()))
        access = (1 if 'public' in modifiers or unit.get('implicit_public') else
                  4 if 'protected' in modifiers else 2 if 'private' in modifiers else 0)
        declared_superclass = unit.get('superclass', '')
        superclass = (_resolve(declared_superclass, unit, known) if declared_superclass else
                      'java.lang.Enum' if kind == 'enum' else 'java.lang.Record' if kind == 'record' else
                      'java.lang.Object' if kind != 'interface' and owner != 'java.lang.Object' else '')
        interfaces = [_resolve(value, unit, known) for value in unit.get('interfaces', ())]
        complete = (not declared_superclass or bool(superclass)) and all(interfaces)
        if not complete:
            rejected.append({'path': unit['path'], 'name': owner, 'reason': 'unresolved_parent_type'})
        symbols[owner] = {
            "kind": "CLASS", "name": owner.rsplit(".", 1)[-1], "owner": owner,
            "access": access | (8 if unit['static'] else 0) |
                      (0x10 if 'final' in modifiers or kind == 'record' else 0) |
                      (0x4000 if kind == 'enum' else 0) |
                      (0x200 if kind == "interface" else 0) |
                      (0x400 if kind == "interface" or
                       'abstract' in modifiers else 0),
            "superclass": superclass, 'hierarchy_complete': complete,
            'method_contract_complete': True,
            'enclosing_instance_type': unit.get('enclosing_owner', '') if not unit['static'] else '',
            'enclosing_owner': unit.get('enclosing_owner', ''),
            "interfaces": [value for value in interfaces if value],
            'package': unit.get('package', owner.rpartition('.')[0]),
            "side": unit["side"], "source_hash": unit["source_sha256"],
            "source_path": unit["path"], "provenance": "materialized_source",
        }
    all_known = known | set(symbols)
    for unit in units:
        kind = unit["kind"]
        if kind not in {"method", "constructor", "field"}:
            continue
        owner = unit["owner"]
        if owner not in symbols:
            continue
        member = unit["name"]
        if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", member):
            if kind == 'method':
                symbols[owner]['method_contract_complete'] = False
            rejected.append({"path": unit["path"], "name": member, "reason": "ambiguous_name"})
            continue
        parameters = []
        for param in unit["parameters"]:
            type_ = _resolve(param["type"], unit, all_known)
            if not type_:
                if kind == 'method':
                    symbols[owner]['method_contract_complete'] = False
                rejected.append({"path": unit["path"], "name": member,
                                 "reason": "unresolved_parameter_type", "type": param["type"]})
                break
            parameters.append({"name": param["name"], "type": type_})
        else:
            return_type = (owner if kind == "constructor" else
                           _resolve(unit["return_type"], unit, all_known))
            if not return_type:
                if kind == 'method':
                    symbols[owner]['method_contract_complete'] = False
                rejected.append({"path": unit["path"], "name": member,
                                 "reason": "unresolved_return_type", "type": unit["return_type"]})
                continue
            modifiers = set(unit.get('modifiers', ()))
            access = (1 if 'public' in modifiers or unit.get('implicit_public') else
                      4 if 'protected' in modifiers else 2 if 'private' in modifiers else 0)
            access |= (8 if unit['static'] else 0) | (0x10 if 'final' in modifiers or unit.get('implicit_final') else 0)
            if (kind == 'method' and not unit.get('has_body', True)
                    and ('abstract' in modifiers or symbols[owner]['access'] & 0x200)):
                access |= 0x400
            throws = [_resolve(value, unit, all_known) for value in unit.get('throws', ())]
            if any(not value for value in throws):
                if kind == 'method':
                    symbols[owner]['method_contract_complete'] = False
                rejected.append({'path': unit['path'], 'name': member, 'reason': 'unresolved_throws_type'})
                continue
            symbol_id = owner + "#" + member + "(" + ",".join(x["type"] for x in parameters) + ")"
            if symbol_id in symbols:
                raise ValueError("CONSTRUCTION_DUPLICATE_SOURCE_SIGNATURE: " + symbol_id)
            symbols[symbol_id] = {
                "kind": kind.upper(), "owner": owner, "name": member,
                "parameters": parameters, "return_type": return_type,
                "access": access, "static": unit["static"], "side": unit["side"],
                'package': symbols[owner]['package'], 'throws': throws,
                'enclosing_instance_type': symbols[owner].get('enclosing_instance_type', '') if kind == 'constructor' else '',
                "source_path": unit["path"], "source_hash": unit["source_sha256"],
                "provenance": "materialized_source",
            }
    for unit in units:
        owner = unit['owner']
        if unit['kind'] != 'class' or any(
            other['owner'] == owner and other['kind'] == 'constructor' for other in units):
            continue
        key = owner + '#<init>()'
        row = symbols[owner]
        symbols[key] = {**row, 'kind': 'CONSTRUCTOR', 'name': '<init>',
                        'parameters': [], 'return_type': owner, 'static': False,
                        'access': row['access'] & 7, 'throws': [],
                        'implicit_default_constructor': True}
    return symbols, tuple(rejected)


def merge_materialized_source_sites(source_pages, target_symbols):
    """Make hash-inspected already-generated Java methods available to hook routing.

    A symbol index assembled before the content shard cannot know about newly
    generated, installed GUI/entity source. Those real project methods are the
    possible original occurrence producers for a custom Mixin. Only parsed,
    hash-checked materialized declarations may join the target ABI, and an
    existing owner is never overwritten.
    """
    if not source_pages:
        return dict(target_symbols)
    if not isinstance(source_pages, Mapping):
        raise ValueError("CONSTRUCTION_SOURCE_PAGES_INVALID")
    original = dict(target_symbols)
    classes = {
        str(row.get("owner"))
        for row in original.values()
        if isinstance(row, Mapping) and row.get("kind") == "CLASS"
    }
    # Re-index newly generated project owners only. Existing target owners
    # retain their verified bytecode contracts, even when source was also
    # present in the project snapshot.
    filtered = {}
    for path, entries in source_pages.items():
        kept = [
            row for row in entries
            if isinstance(row, Mapping) and row.get("owner") not in classes
        ]
        if kept:
            filtered[path] = kept
    if not filtered:
        return original
    parsed, rejected = inspected_project_source_symbols(filtered, original)
    if rejected:
        # Partially parsed Java is not a valid producer. A source hook may only
        # target an owner whose method signatures were resolved completely.
        invalid = {row.get("name") for row in rejected}
        incomplete = {
            row.get("owner") for row in parsed.values()
            if isinstance(row, Mapping) and row.get("kind") == "CLASS"
            and not row.get("method_contract_complete", False)
        }
        parsed = {
            key: row for key, row in parsed.items()
            if row.get("owner") not in incomplete
            and row.get("owner") not in invalid
        }
    original.update(parsed)
    return original
