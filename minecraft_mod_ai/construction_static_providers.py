"""Ground static field inputs using target classpath facts, not guessed names."""
from __future__ import annotations

from .construction_operation_contracts import result_fits
from .construction_retrieval import _words
from .rag_index import _meaningful_terms


def _terms(value):
    return set(_meaningful_terms(_words(value)))


def _required_type(consumers):
    if not consumers:
        return None
    expected = {row['expected_type'] for row in consumers if row.get('expected_type')}
    if len(expected) != 1 or any(row.get('allowed_types') or row.get('expected_types')
                                 or row.get('expected_element_type') for row in consumers):
        return None
    type_ = next(iter(expected))
    if type_ in {'java.lang.String', 'java.lang.Object', 'void',
                'boolean', 'byte', 'char', 'short', 'int', 'long', 'float', 'double'}:
        return None
    return type_


def _candidate_score(key, row, *, index, consumers, purpose_terms):
    """Use exact inspected field type first, semantic owner/name evidence second."""
    if (row.get('kind') != 'FIELD' or row.get('provenance') == 'construction_plan'
            or not (row.get('static') or (row.get('access', 0) & 8))):
        return None
    name_terms = _terms(row.get('name', ''))
    if len(name_terms) < 2 or not name_terms <= purpose_terms:
        return None
    if not index.accessible(key) or not result_fits(row['return_type'], consumers, index):
        return None
    owner_terms = _terms(row['owner'].rsplit('.', 1)[-1])
    role_match = len({term[:5] for term in owner_terms if len(term) >= 5}
                     & {term[:5] for term in purpose_terms if len(term) >= 5})
    return len(name_terms) * 10 + role_match


def preferred_static_field(symbols, evidence_index, consumers, purpose, *, excluded=()):
    """Return one unambiguous, type-correct field symbol, otherwise abstain."""
    required = _required_type(consumers)
    if evidence_index is None or required is None:
        return None
    cache = getattr(evidence_index, '_static_provider_cache', None)
    if cache is None:
        cache = {}
        evidence_index._static_provider_cache = cache
    cache_key = (required, purpose, tuple(sorted(excluded)))
    if cache_key in cache:
        return cache[cache_key]
    purpose_terms = _terms(purpose)
    choices = []
    excluded_set = set(excluded)
    for key, row in symbols.items():
        if key in excluded_set:
            continue
        score = _candidate_score(key, row, index=evidence_index,
                                 consumers=consumers, purpose_terms=purpose_terms)
        if score is not None:
            choices.append((score, key))
    choices.sort(key=lambda item: (-item[0], item[1]))
    result = (choices[0][1] if choices and
              (len(choices) == 1 or choices[0][0] > choices[1][0]) else None)
    cache[cache_key] = result
    return result
