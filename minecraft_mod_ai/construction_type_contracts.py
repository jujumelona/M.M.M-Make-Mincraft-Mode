"""Mechanical type identities and inherited contracts from inspected evidence."""
from __future__ import annotations

from collections import defaultdict


def canonical_type_name(value):
    return str(value).replace('/', '.').replace('$', '.')


def owner_package(symbols, owner):
    return symbols.get(owner, {}).get('package', owner.rpartition('.')[0])


def inspected_subtype(symbols, actual, expected):
    if expected == 'java.lang.Object' and (actual.endswith('[]') or symbols.get(actual, {}).get('kind') == 'CLASS'):
        return True
    if actual.endswith('[]'):
        if expected in {'java.lang.Cloneable', 'java.io.Serializable'}:
            return True
        if expected.endswith('[]'):
            return actual == expected or inspected_subtype(symbols, actual[:-2], expected[:-2])
        return False
    pending, seen = [actual], set()
    while pending:
        current = pending.pop()
        if current == expected:
            return True
        if current in seen:
            continue
        seen.add(current)
        row = symbols.get(current, {})
        pending.extend(value for value in (row.get('superclass'), *row.get('interfaces', ())) if value)
    return False


def annotate_functional_interfaces(symbols):
    methods = defaultdict(list)
    for key, row in symbols.items():
        if (row.get('kind') == 'METHOD' and not row.get('static')
                and not row.get('access', 0) & (0x0002 | 0x0040 | 0x1000)):
            parameters = tuple(p['type'] if isinstance(p, dict) else p for p in row.get('parameters', ()))
            methods[row['owner']].append((key, row, (row['name'], parameters)))
    object_methods = {signature for _, row, signature in methods.get('java.lang.Object', ())
                      if row.get('access', 0) & 1}
    for owner, symbol in symbols.items():
        if symbol.get('kind') != 'CLASS' or not symbol.get('access', 0) & 0x0200:
            continue
        for field in ('functional_parameters', 'functional_return_type', 'functional_throws',
                      'functional_contract_sources'):
            symbol.pop(field, None)
        pending, ancestors, complete = [owner], set(), True
        while pending:
            current = pending.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            row = symbols.get(current, {})
            if (row.get('kind') != 'CLASS' or row.get('hierarchy_complete') is False
                    or row.get('method_contract_complete') is False):
                complete = False
                break
            pending.extend(row.get('interfaces', ()))
        if not complete:
            continue
        by_signature = defaultdict(list)
        for ancestor in ancestors:
            for key, row, signature in methods.get(ancestor, ()):
                if signature not in object_methods:
                    by_signature[signature].append((key, row))
        abstract = {}
        for signature, declarations in by_signature.items():
            most_specific = [(key, row) for key, row in declarations if not any(
                other['owner'] != row['owner']
                and inspected_subtype(symbols, other['owner'], row['owner'])
                for _, other in declarations)]
            if any(not row.get('access', 0) & 0x0400 for _, row in most_specific):
                continue
            abstract[signature] = most_specific
        if len(abstract) != 1:
            continue
        (_, parameters), declarations = next(iter(abstract.items()))
        returns = {row['return_type'] for _, row in declarations}
        compatible = [type_ for type_ in returns if all(
            type_ == other or inspected_subtype(symbols, type_, other) for other in returns)]
        if len(compatible) != 1:
            continue
        throws = {type_ for _, row in declarations for type_ in row.get('throws', ())}
        symbol['functional_parameters'] = list(parameters)
        symbol['functional_return_type'] = compatible[0]
        symbol['functional_throws'] = sorted(type_ for type_ in throws if all(
            any(inspected_subtype(symbols, type_, allowed) for allowed in row.get('throws', ()))
            for _, row in declarations))
        symbol['functional_contract_sources'] = sorted(key for key, _ in declarations)
