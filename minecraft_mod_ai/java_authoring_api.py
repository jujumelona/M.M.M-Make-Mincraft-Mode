"""Small Java interface views preserve inspected type dependencies and exact ABI."""
from __future__ import annotations

import re
from collections.abc import Mapping

from .construction_type_contracts import canonical_type_name


_ABI = (
    "kind", "owner", "name", "static", "access", "return_type", "parameters",
    "varargs", "generic_signature", "signature", "enclosing_instance_type",
    "generic_parameters", "generic_return_type", "type_parameters", "throws",
    "generic_throws", "superclass", "interfaces", "generic_superclass", "generic_interfaces",
    "enclosing_owner", "functional_parameters", "functional_return_type", "functional_throws",
)


def _member_dependencies(members):
    names = set()
    for row in members:
        types = [row.get('return_type', ''), *row.get('parameters', ()),
                 row.get('generic_return_type', ''), *row.get('generic_parameters', ())]
        for type_ in types:
            names.update(canonical_type_name(name) for name in re.findall(
                r'[A-Za-z_$][\w$]*(?:[./][A-Za-z_$][\w$]*)*', str(type_)))
    return names


def _selected_members(entry, full_owner, selected):
    return [row for row in entry.get('members', ()) if isinstance(row, dict)
            and (full_owner or (row.get('owner'), row.get('kind'), row.get('name')) in selected)]


def _nested_owners(records, owner):
    return {name for name, nested in records.items()
            if canonical_type_name((nested.get('declaration') or {}).get('enclosing_owner', '')) == owner}


def _class_records(evidence):
    return {row.get('verified_class'): row for row in evidence.get('class_members', ())
            if isinstance(row, dict) and isinstance(row.get('verified_class'), str)}


def scoped_body_api_records(evidence):
    """Keep selected calls and the callable contracts of selected class types.

    A CLASS anchor is not a method named after the class. Filtering its
    members by (owner, CLASS, name) erased every actual method and all builder
    dependencies. Follow only inspected type edges within this bounded view.
    """
    selected = {(row.get('owner'), row.get('kind'), row.get('name'))
                for row in evidence.get('target_api', ()) if isinstance(row, dict)}
    full_owners = {owner for owner, kind, _ in selected if kind == 'CLASS'}
    records = _class_records(evidence)
    pending = sorted({owner for owner, _, _ in selected})
    answer, seen = {}, set()
    while pending:
        owner = pending.pop(0)
        mode = owner in full_owners
        if (owner, mode) in seen or owner not in records:
            continue
        seen.add((owner, mode))
        entry = records[owner]
        members = _selected_members(entry, mode, selected)
        if not members and not mode:
            continue
        answer[owner] = {**entry, 'members': members, 'body_context_selects_only_target_api': True}
        dependencies = _member_dependencies(members)
        if mode:
            # Static nested factories are accessed as types, even when no
            # root method returns them. Use inspected enclosure, not names.
            dependencies.update(_nested_owners(records, owner))
        for name in sorted(dependencies & records.keys() - full_owners):
            full_owners.add(name)
            pending.append(name)
    return [answer[name] for name in sorted(answer)]


def _present_abi_fields(record):
    optional = {"type_parameters", "throws", "generic_throws", "interfaces", "generic_interfaces",
                "generic_parameters"}
    return ((key, record[key]) for key in _ABI if key in record and record[key] != ""
            and record[key] is not None and not (key in optional and record[key] == []))


def _abi(record):
    return dict(_present_abi_fields(record))


def target_interfaces(evidence):
    """Strip index metadata after following selected call and type dependencies."""
    selected = [row for row in evidence.get("target_api", evidence.get("selected_target_api", ()))
                if isinstance(row, Mapping)]
    records = (scoped_body_api_records({**evidence, "target_api": selected})
               if selected else evidence.get("class_members", ()))
    return {"target_api": [_abi(row) for row in selected], "class_members": [
        {"verified_class": row.get("verified_class"),
         "declaration": _abi(row.get("declaration") or {}),
         "members": [_abi(member) for member in row.get("members", ())]}
        for row in records if isinstance(row, Mapping)],
        "typed_input_producers": evidence.get("typed_input_producers", [])}
