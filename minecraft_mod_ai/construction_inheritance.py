"""Expose inspected inherited APIs through accessible public receiver types."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from .construction_type_contracts import owner_package


def project_inherited_api(symbols):
    classes = {key: row for key, row in symbols.items() if row.get('kind') == 'CLASS'}
    members = defaultdict(list)
    for key, row in list(symbols.items()):
        if row.get('kind') in {'METHOD', 'FIELD'} and not row.get('inherited_from'):
            members[row['owner']].append((key, row))

    def public_type(owner):
        seen = set()
        while owner:
            if owner in seen or owner not in classes or not classes[owner].get('access', 0) & 1:
                return False
            seen.add(owner)
            owner = classes[owner].get('enclosing_owner', '')
        return True

    def signature(row):
        if row['kind'] == 'FIELD':
            return 'FIELD', row['name']
        parameters = tuple(value['type'] if isinstance(value, dict) else value for value in row.get('parameters', ()))
        return 'METHOD', row['name'], parameters

    for receiver, definition in classes.items():
        if not public_type(receiver):
            continue
        blocked = {signature(row) for _key, row in members.get(receiver, ())}
        parents = [definition.get('superclass', ''), *definition.get('interfaces', ())]
        frontier, seen = [parent for parent in parents if parent], {receiver}
        while frontier:
            ancestor = frontier.pop(0)
            if ancestor in seen:
                continue
            seen.add(ancestor)
            row = classes.get(ancestor, {})
            inherited = []
            for key, member in members.get(ancestor, ()):
                shape = signature(member)
                if shape in blocked or not member.get('access', 0) & 5:
                    continue
                # Interface static methods are never inherited by a child type.
                if row.get('access', 0) & 0x0200 and member.get('static'):
                    continue
                inherited.append(shape)
                if public_type(ancestor):
                    continue
                suffix = key.split('#', 1)[1]
                projected_key = receiver + '#' + suffix
                if projected_key in symbols:
                    continue
                projection = deepcopy(member)
                projection.update(owner=receiver, package=owner_package(symbols, receiver),
                                  inherited_from=key, declaring_owner=ancestor,
                                  receiver_source_hash=definition.get('source_hash', ''),
                                  receiver_source_path=definition.get('source_path', ''))
                if definition.get('side') == 'CLIENT':
                    projection['side'] = 'CLIENT'
                symbols[projected_key] = projection
            blocked.update(inherited)
            frontier.extend(parent for parent in (row.get('superclass'), *row.get('interfaces', ())) if parent)
