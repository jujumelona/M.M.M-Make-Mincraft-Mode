"""Expose inspected inherited APIs through accessible public receiver types."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from .construction_type_contracts import owner_package, inspected_subtype


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

    def field_members(receiver, name):
        # Resolve each direct parent separately before combining its fields.
        # A superclass declaration may hide its own ancestor/interface fields;
        # a separately implemented interface can still make the child ambiguous.
        resolved, visiting, pending = {}, set(), [(receiver, False)]
        while pending:
            owner, finish = pending.pop()
            if owner in resolved:
                continue
            own = [(key, row) for key, row in members.get(owner, ())
                   if row['kind'] == 'FIELD' and row['name'] == name]
            if own:
                resolved[owner] = own
                continue
            row = classes.get(owner, {})
            parents = [value for value in (row.get('superclass'), *row.get('interfaces', ())) if value]
            if finish:
                combined = {key: value for parent in parents for key, value in resolved[parent]}
                resolved[owner] = list(combined.items())
                visiting.remove(owner)
            else:
                if owner in visiting:
                    raise ValueError('CONSTRUCTION_INHERITANCE_CYCLE')
                visiting.add(owner)
                pending.append((owner, True))
                pending.extend((parent, False) for parent in reversed(parents) if parent not in resolved)
        return resolved[receiver]

    for receiver, definition in classes.items():
        if not public_type(receiver):
            continue
        own = {signature(row) for _key, row in members.get(receiver, ())}
        frontier = [parent for parent in (definition.get('superclass'), *definition.get('interfaces', ())) if parent]
        ancestors, seen = [], {receiver}
        while frontier:
            ancestor = frontier.pop()
            if ancestor in seen:
                continue
            seen.add(ancestor)
            ancestors.append(ancestor)
            row = classes.get(ancestor, {})
            frontier.extend(parent for parent in (row.get('superclass'), *row.get('interfaces', ())) if parent)
        if all(public_type(ancestor) for ancestor in ancestors):
            continue
        class_depth, current = {}, definition.get('superclass', '')
        while current and current not in class_depth:
            class_depth[current] = len(class_depth)
            current = classes.get(current, {}).get('superclass', '')
        groups = defaultdict(list)
        for ancestor in ancestors:
            for key, member in members.get(ancestor, ()):
                shape = signature(member)
                if shape in own:
                    continue
                if member['kind'] == 'METHOD':
                    if not member.get('access', 0) & 5:
                        continue
                    if classes.get(ancestor, {}).get('access', 0) & 0x0200 and member.get('static'):
                        continue
                # Even a private nearer field hides a deeper public field.
                groups[shape].append((key, member))
        for shape, candidates in groups.items():
            class_members = [(key, row) for key, row in candidates if row['owner'] in class_depth]
            if shape[0] == 'FIELD':
                candidates = field_members(receiver, shape[1])
            elif class_members:
                nearest = min(class_depth[row['owner']] for _key, row in class_members)
                candidates = [(key, row) for key, row in class_members if class_depth[row['owner']] == nearest]
            else:
                candidates = [(key, row) for key, row in candidates if not any(
                    other['owner'] != row['owner'] and inspected_subtype(symbols, other['owner'], row['owner'])
                    for _other_key, other in candidates)]
            if shape[0] == 'FIELD':
                if len(candidates) != 1 or not candidates[0][1].get('access', 0) & 5:
                    continue
            else:
                result_types = {row['return_type'] for _key, row in candidates}
                most_specific = [type_ for type_ in result_types if all(
                    type_ == other or inspected_subtype(symbols, type_, other) for other in result_types)]
                if len(most_specific) != 1:
                    continue
                candidates.sort(key=lambda item: (item[1]['return_type'] != most_specific[0], item[0]))
            key, member = candidates[0]
            if public_type(member['owner']):
                continue
            projected_key = receiver + '#' + key.split('#', 1)[1]
            if projected_key in symbols:
                continue
            projection = deepcopy(member)
            projection.update(owner=receiver, package=owner_package(symbols, receiver),
                              inherited_from=key, declaring_owner=member['owner'],
                              inherited_contract_sources=[item[0] for item in candidates],
                              receiver_source_hash=definition.get('source_hash', ''),
                              receiver_source_path=definition.get('source_path', ''))
            if shape[0] == 'METHOD':
                throws = {type_ for _key, row in candidates for type_ in row.get('throws', ())}
                projection['throws'] = sorted(type_ for type_ in throws if all(
                    any(inspected_subtype(symbols, type_, allowed) for allowed in row.get('throws', ()))
                    for _key, row in candidates))
            if definition.get('side') == 'CLIENT':
                projection['side'] = 'CLIENT'
            symbols[projected_key] = projection
