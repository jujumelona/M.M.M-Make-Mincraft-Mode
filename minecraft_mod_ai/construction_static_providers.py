"""Ground static field inputs using target classpath facts, not guessed names."""
from __future__ import annotations

from .construction_operation_contracts import result_fits
from .construction_retrieval import _words
from .rag_index import _meaningful_terms


def _terms(value):
    return set(_meaningful_terms(_words(value)))


def preferred_static_field(symbols, evidence_index, consumers, purpose, *, excluded=()):
    """Return one uniquely supported field symbol, otherwise abstain."""
    if evidence_index is None or not consumers:
        return None
    expected = {row['expected_type'] for row in consumers if row.get('expected_type')}
    if len(expected) != 1 or any(row.get('allowed_types') or row.get('expected_types')
                                 or row.get('expected_element_type') for row in consumers):
        return None
    expected_type = next(iter(expected))
    if expected_type in {'java.lang.String', 'java.lang.Object', 'void',
                         'boolean', 'byte', 'char', 'short', 'int', 'long', 'float', 'double'}:
        return None
    purpose_terms = _terms(purpose)
    if not purpose_terms:
        return None
    excluded = set(excluded)
    choices = []
    for key, row in symbols.items():
        if (row.get('kind') != 'FIELD' or key in excluded
                or row.get('provenance') == 'construction_plan'
                or not (row.get('static') or (row.get('access', 0) & 8))):
            continue
        name_terms = _terms(row.get('name', ''))
        if len(name_terms) < 2 or not name_terms <= purpose_terms:
            continue
        if not evidence_index.accessible(key) or not result_fits(row['return_type'], consumers, evidence_index):
            continue
        owner_terms = _terms(row['owner'].rsplit('.', 1)[-1])
        role_match = len({term[:5] for term in owner_terms if len(term) >= 5}
                         & {term[:5] for term in purpose_terms if len(term) >= 5})
        choices.append((len(name_terms) * 10 + role_match, key))
    if not choices:
        return None
    choices.sort(key=lambda item: (-item[0], item[1]))
    if len(choices) > 1 and choices[0][0] == choices[1][0]:
        return None
    return choices[0][1]
