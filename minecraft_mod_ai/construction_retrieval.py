"""Purpose-driven discovery of real target APIs and complete source evidence.

Names are returned evidence, never a prerequisite invented by the author.
The target inventory remains authoritative; old source is porting evidence.
"""
from __future__ import annotations
from .construction_type_contracts import owner_package, class_accessible

import hashlib
import json
import re
from collections import Counter, defaultdict
from bisect import bisect_right
from dataclasses import replace

from .construction_state import PAGE_SIZE, page_result, symbol_query_key, symbol_record
from .java_operation_ir import parameter_types, PRIMITIVES, WIDENING, BOXING
from .rag_index import RAGChunk, rank_context_chunks, _lexical_score, _meaningful_terms, _tokens


def _words(text):
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(text))
    return re.sub(r'[.$#/_:<>(),;\[\]-]+', ' ', text)


def _name_parts(row):
    """Callable-name words are different evidence from an owner's type name."""
    return tuple(_meaningful_terms(_words(row.get('name', ''))))


def _purpose_name_weights(purpose):
    """Use matching, inspected method names to bridge inflected goal terms.

    Morphological prefixes nominate *inspected* callables, never guessed APIs.
    Action-noun/gerund terms weigh more than incidental type nouns.
    """
    weights = {}
    for term in _meaningful_terms(_words(purpose)):
        if len(term) < 5:
            continue
        prefix = term[:5]
        weight = 4 if term.endswith(('tion', 'sion', 'ment', 'ing')) else 1
        weights[prefix] = max(weight, weights.get(prefix, 0))
    return weights


def _explicit_method_prefix(purpose):
    """A dotted callable is a search *intent*, never proof that an API exists.

    In particular, a model-suggested EntityType.registerEntity must find
    actual `register*` members in the inspected classpath, not every method
    whose owner mentions Entity, Type, or Value.
    """
    matches = re.findall(r'\b([A-Z][A-Za-z0-9_$]*)\.([a-z][A-Za-z0-9_$]*)\b', purpose)
    if not matches:
        return None
    name = matches[0][1]
    parts = _name_parts({'name': name})
    return parts[0][:5] if parts and len(parts[0]) >= 5 else None


def _owner_name_precision(owner, query_terms):
    """Match the actual declaring class role, not its package path."""
    name = owner.rsplit('.', 1)[-1].replace('$', '.')
    parts = set(_meaningful_terms(_words(name)))
    return len(parts & query_terms) / len(parts) if parts else 0.0


def _description(row):
    return _words(' '.join(str(value) for value in (
        row.get('owner', ''), row.get('name', ''), row.get('kind', ''),
        row.get('purpose', ''), row.get('return_type', ''),
        *parameter_types(row), row.get('superclass', ''), *row.get('interfaces', ()),
        *row.get('functional_parameters', ()), row.get('functional_return_type', ''),
    )))


def _selection_contract(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'contracts', 'excluded_symbols', 'constructor_result'}
            or not isinstance(value['contracts'], list) or not isinstance(value['excluded_symbols'], list)
            or any(not isinstance(key, str) for key in value['excluded_symbols'])
            or type(value['constructor_result']) is not bool):
        raise ValueError('CONSTRUCTION_RETRIEVAL_SELECTION_INVALID')
    scalar = {'mode', 'expected_type', 'expected_element_type', 'required_shape'}
    sequences = {'expected_types', 'allowed_types'}
    for row in value['contracts']:
        if (not isinstance(row, dict) or set(row) - scalar - sequences
                or row.get('mode') not in {'expression', 'statement'}
                or any(not isinstance(row[key], str) for key in scalar & row.keys())
                or any(not isinstance(row[key], list) or any(not isinstance(item, str) for item in row[key])
                       for key in sequences & row.keys())):
            raise ValueError('CONSTRUCTION_RETRIEVAL_SELECTION_INVALID')
    return {'contracts': [dict(row) for row in value['contracts']],
            'excluded_symbols': sorted(set(value['excluded_symbols'])),
            'constructor_result': value['constructor_result']}


class ConstructionEvidenceIndex:
    def __init__(self, symbols, source_pages, router, *, target_path, owner, authoring_contract):
        self.symbols, self.router = symbols, router
        self.target_path, self.owner = target_path, owner
        self.contract = authoring_contract
        self.goals = {}
        # Keep one ranked inventory, not one full inventory per graph node.
        # Older goals remain reconstructible from their immutable selectors.
        self.ordering_goal = None
        self.ordering = None
        self.active_eligible = {}
        self.by_owner = defaultdict(list)
        # Postings contain owners, not duplicate copies of the entire
        # classpath index. Query-local scoring inspects only matching owners.
        self.member_name_owners = defaultdict(set)
        self.member_name_keys = defaultdict(set)
        self.name_weights = {}
        self.strong_name_prefixes = frozenset()
        self.exact_action_available = False
        self.aliases = defaultdict(set)
        self.member_text = {}
        self.member_prepared = {}
        self.scope_children = defaultdict(set)
        self.scope_owners = defaultdict(set)
        self.owner_scopes = {}
        self.eligible = {}
        self.eligibility_header = None
        self.selection_eligible = None
        self.member_orderings = {}
        self.owner_term_counts = defaultdict(lambda: defaultdict(Counter))
        self.owner_document_counts = defaultdict(Counter)
        self.own_documents = {}
        self.owner_chunks = {}
        self.owner_prepared = {}
        for key, row in symbols.items():
            if row.get('provenance') == 'construction_plan' and row['owner'] != owner:
                continue
            self.by_owner[row['owner']].append(key)
            self._record_document(key, row)
            self._index_name_owner(key, row)
            if row.get('kind') == 'CLASS':
                self.aliases[symbol_query_key(key)].add(key)
                self.aliases[symbol_query_key(row['name'])].add(key)
        for type_owner in self.by_owner:
            if type_owner not in self.owner_chunks:
                text = _words(type_owner)
                self.owner_chunks[type_owner] = RAGChunk(type_owner, type_owner, text, 1, 1, '', {})
                self.owner_prepared[type_owner] = (frozenset(_tokens(text)), text.casefold())
            self._add_owner_scope(type_owner)
        self.sources = {}
        for source_ref, units in source_pages.items():
            for offset, unit in enumerate(units):
                if unit.get('side') == 'CLIENT' and not target_path.startswith('src/client/'):
                    continue
                key = source_ref + ':' + str(offset)
                self.sources[key] = (source_ref, offset, unit)
        self.source_chunks = tuple(
            RAGChunk(key, ref, _words(unit.get('signature', '') + '\n' +
                     ('' if unit.get('kind') in {'class', 'interface', 'enum', 'record'} else unit.get('code', ''))),
                     unit.get('start_line', 1), unit.get('end_line', 1),
                     unit.get('source_sha256', ''), {})
            for key, (ref, _offset, unit) in self.sources.items())
        self.source_targets = {key: self._source_targets(unit)
                               for key, (_ref, _offset, unit) in self.sources.items()}
        self.source_prepared = {row.chunk_id: (frozenset(_tokens(row.text)), row.text.casefold())
                                for row in self.source_chunks}
        self.dense = self._dense_allowed()
        if self.dense and self.source_chunks:
            vectors = router.embed([row.text for row in self.source_chunks])
            if len(vectors) != len(self.source_chunks) or any(not vector for vector in vectors):
                raise ValueError('CONSTRUCTION_RETRIEVAL_EMBEDDING_SHAPE')
            self.source_chunks = tuple(replace(row, embedding=tuple(vector))
                                       for row, vector in zip(self.source_chunks, vectors))
        self.query_vectors = {}
        self.source_ranking_query = None
        self.source_ranking = None

    def _index_name_owner(self, key, row):
        if row.get('kind') == 'METHOD':
            for part in _name_parts(row):
                if len(part) >= 5:
                    self.member_name_owners[part[:5]].add(row['owner'])
                    self.member_name_keys[part[:5]].add(key)

    def _record_document(self, key, row):
        owner, kind = row['owner'], row['kind']
        text = _description(row)
        terms = frozenset(_tokens(text))
        if key in self.own_documents:
            previous_kind, previous_terms = self.own_documents[key]
            self.owner_term_counts[owner][previous_kind].subtract(previous_terms)
            self.owner_document_counts[owner][previous_kind] -= 1
        self.owner_term_counts[owner][kind].update(terms)
        self.owner_document_counts[owner][kind] += 1
        if owner == self.owner:
            self.own_documents[key] = (kind, terms)
        if kind == 'CLASS':
            self.owner_chunks[owner] = RAGChunk(owner, owner, text, 1, 1, '', {})
            self.owner_prepared[owner] = (terms, text.casefold())

    def include_source(self, source_ref, units):
        """Publish accepted sibling evidence without promoting it to bytecode."""
        additions = []
        for offset, unit in enumerate(units):
            key = source_ref + ':' + str(offset)
            if key in self.sources:
                raise ValueError('CONSTRUCTION_SOURCE_EVIDENCE_DUPLICATE')
            if unit.get('side') == 'CLIENT' and not self.target_path.startswith('src/client/'):
                continue
            self.sources[key] = (source_ref, offset, unit)
            text = _words(unit.get('signature', '') + '\n' + unit.get('code', ''))
            additions.append(RAGChunk(key, source_ref, text, 1, 1, unit.get('source_sha256', ''), {}))
            self.source_targets[key] = self._source_targets(unit)
            self.source_prepared[key] = (frozenset(_tokens(text)), text.casefold())
        if self.dense and additions:
            vectors = self.router.embed([row.text for row in additions])
            if len(vectors) != len(additions) or any(not vector for vector in vectors):
                raise ValueError('CONSTRUCTION_RETRIEVAL_EMBEDDING_SHAPE')
            additions = [replace(row, embedding=tuple(vector)) for row, vector in zip(additions, vectors)]
        self.source_chunks = (*self.source_chunks, *additions)
        self.source_ranking_query = None
        self.ordering_goal = None

    def _add_owner_scope(self, owner):
        package = owner_package(self.symbols, owner)
        scopes = ['package:']
        parts = package.split('.') if package else []
        for index in range(1, len(parts) + 1):
            scope = 'package:' + '.'.join(parts[:index])
            self.scope_children[scopes[-1]].add(scope)
            scopes.append(scope)
        # '$' normalization does not turn an enclosing class into a package.
        enclosing, seen, types = owner, set(), []
        while enclosing:
            if enclosing in seen:
                raise ValueError('CONSTRUCTION_ENCLOSING_TYPE_CYCLE')
            seen.add(enclosing)
            types.append('type:' + enclosing)
            enclosing = self.symbols.get(enclosing, {}).get('enclosing_owner', '')
        for leaf in reversed(types):
            self.scope_children[scopes[-1]].add(leaf)
            scopes.append(leaf)
        self.owner_scopes[owner] = tuple(scopes)
        for scope in scopes:
            self.scope_owners[scope].add(owner)

    def include_symbol(self, key):
        # Declaration authors add only their own real planned signature. This
        # incremental update avoids rescanning the immutable classpath map.
        row = self.symbols[key]
        if row['owner'] != self.owner or row.get('provenance') != 'construction_plan':
            raise ValueError('CONSTRUCTION_RETRIEVAL_LOCAL_AUTHORITY')
        if key not in self.by_owner[self.owner]:
            self.by_owner[self.owner].append(key)
        self._record_document(key, row)
        self._index_name_owner(key, row)
        self.member_text[key] = _description(row)
        self.member_prepared[key] = (frozenset(_tokens(self.member_text[key])), self.member_text[key].casefold())
        self.eligible.clear()
        self.selection_eligible = None
        self.active_eligible = {}
        self.ordering_goal = None

    def _member_name_score(self, key):
        parts = _name_parts(self.symbols[key])
        if not parts:
            return 0.0
        matches = sum(self.name_weights.get(part[:5], 0)
                      for part in parts if len(part) >= 5)
        return matches / (1.0 + 0.65 * (len(parts) - 1))

    def intent_compatible(self, key):
        """Abstain from unrelated call names when an exact action API exists.

        No exact action-name evidence means the author retains normal discovery.
        """
        if not self.exact_action_available or not self.strong_name_prefixes:
            return True
        parts = _name_parts(self.symbols[key])
        if not any(part[:5] in self.strong_name_prefixes for part in parts):
            return False
        return all(part[:5] in self.strong_name_prefixes or part in self.query_terms
                   for part in parts)

    def _member_lexical_score(self, key):
        # Memoize only members of displayed owners. Duplicating every full
        # symbol's text/token set per concurrent artifact wastes host memory.
        if key not in self.member_text:
            text = _description(self.symbols[key])
            self.member_text[key] = text
            self.member_prepared[key] = (frozenset(_tokens(text)), text.casefold())
        return _lexical_score(self.query_terms, self.query_lower, self.member_text[key],
                              prepared=self.member_prepared[key])

    def _goal(self, descriptor):
        goal = 'intent:' + hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:24]
        self.goals[goal] = descriptor
        return goal

    def _ensure(self, goal):
        if goal not in self.goals:
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOAL_MISSING')
        row = self.goals[goal]
        base = self._goal({**row, 'scope': 'package:'})
        if base != self.ordering_goal:
            self.prepare(row['purpose'], kinds=row['kinds'], selection=row.get('selection'))
        return row

    def root_goal(self, goal):
        return self._goal({**self.goals[goal], 'scope': 'package:'})

    def scoped_goal(self, goal, scope):
        if scope not in self.scope_owners:
            raise ValueError('CONSTRUCTION_RETRIEVAL_SCOPE_MISSING')
        return self._goal({**self.goals[goal], 'scope': scope})

    def _eligible_owners(self, kinds, selection=None):
        header = self.symbols.get(self.owner, {})
        fingerprint = (header.get('superclass', ''), tuple(header.get('interfaces', ())))
        if fingerprint != self.eligibility_header:
            self.eligible.clear()
            self.selection_eligible = None
            self.eligibility_header = fingerprint
        kinds = tuple(sorted(kinds))
        if kinds not in self.eligible:
            rows = defaultdict(list)
            for owner, keys in self.by_owner.items():
                for key in keys:
                    row = self.symbols[key]
                    if row.get('kind') in kinds and self.accessible(key, constructor=row.get('kind') == 'CONSTRUCTOR'):
                        rows[owner].append(key)
            self.eligible[kinds] = dict(rows)
        base = self.eligible[kinds]
        if selection is None:
            return base
        cache_key = (kinds, json.dumps(selection, sort_keys=True))
        if self.selection_eligible is not None and self.selection_eligible[0] == cache_key:
            return self.selection_eligible[1]
        from .construction_operation_contracts import result_fits
        excluded = set(selection['excluded_symbols'])
        fits = {}
        rows = {}
        for owner, keys in base.items():
            selected = []
            for key in keys:
                row = self.symbols[key]
                if key in excluded or row.get('inherited_from') in excluded:
                    continue
                type_ = owner if row['kind'] == 'CLASS' or selection['constructor_result'] else row['return_type']
                if type_ not in fits:
                    fits[type_] = result_fits(type_, selection['contracts'], self)
                if fits[type_]:
                    selected.append(key)
            if selected:
                rows[owner] = selected
        # Retain one active filtered view, not another full classpath per slot.
        self.selection_eligible = (cache_key, rows)
        return rows

    def _focused_method_owners(self, prefix, query, selection):
        """Resolve a typed callable hint against a *small* name posting first.

        Avoid building the complete eligible method index merely to score all
        its methods against an explicit callable. This is a search route, not
        permission to invoke a guessed member.
        """
        if not prefix:
            return {}
        from .construction_operation_contracts import result_fits
        excluded = set(selection['excluded_symbols']) if selection else set()
        constraints = selection['contracts'] if selection else ()
        terms = set(_meaningful_terms(_words(query)))
        selected = defaultdict(list)
        for key in self.member_name_keys.get(prefix, ()):
            row = self.symbols[key]
            parts = _name_parts(row)
            if not all(part[:5] == prefix or part in terms for part in parts):
                continue
            if key in excluded or row.get('inherited_from') in excluded:
                continue
            if not self.accessible(key):
                continue
            if constraints and not result_fits(row['return_type'], constraints, self):
                continue
            selected[row['owner']].append(key)
        return dict(selected)

    def _dense_allowed(self):
        registry = getattr(self.router, 'registry', None)
        if registry is None:
            return False
        from .model_adapters.base import ModelConfigurationError
        from .retrieval_cpu_budget_contract import _dense_opted_in
        try:
            config = registry.role(self.router.profile, 'embedding')
        except ModelConfigurationError:
            return False
        device = str(config.extra.get('device', 'cpu') or 'cpu').casefold()
        return config.adapter == 'embedding' and (not device.startswith('cpu') or _dense_opted_in())

    def _subtype(self, actual, expected):
        if actual == expected:
            return True
        if actual in PRIMITIVES or expected in PRIMITIVES:
            return False
        if expected == 'java.lang.Object':
            return True
        if actual.endswith('[]'):
            if expected in {'java.lang.Cloneable', 'java.io.Serializable'}:
                return True
            if expected.endswith('[]'):
                a, b = actual[:-2], expected[:-2]
                return a == b or (a not in PRIMITIVES and b not in PRIMITIVES and self._subtype(a, b))
            return False
        pending, seen = [actual], set()
        while pending:
            current = pending.pop()
            if current == expected:
                return True
            if current in seen:
                continue
            seen.add(current)
            row = self.symbols.get(current, {})
            parents = [row.get('superclass', ''), *row.get('interfaces', ())]
            for parent in parents:
                # Discovery aliases may fold case; Java type identity may not.
                if parent:
                    pending.append(parent)
        return False

    def assignable(self, actual, expected):
        # Same Java conversion rules as the final IR admission check, using
        # actual class parents rather than guessed simple-name compatibility.
        if 'void' in {actual, expected}:
            return False
        if actual == expected or expected in WIDENING.get(actual, ()):
            return True
        if actual in BOXING:
            return self._subtype(BOXING[actual], expected)
        unboxed = {value: key for key, value in BOXING.items()}.get(actual)
        if unboxed and (unboxed == expected or expected in WIDENING.get(unboxed, ())):
            return True
        return self._subtype(actual, expected)

    def accessible(self, key, *, constructor=False):
        row = self.symbols[key]
        if row.get('side') == 'CLIENT' and not self.target_path.startswith('src/client/'):
            return False
        if row.get('provenance') == 'construction_plan' and row['owner'] != self.owner:
            return False
        owner = self.symbols.get(row['owner'], {})
        declaring = row.get('declaring_owner') or row['owner']
        same_package = owner_package(self.symbols, declaring) == owner_package(self.symbols, self.owner)
        if not class_accessible(self.symbols, row['owner'], self.owner, self.target_path):
            return False
        access = row.get('access')
        if access is not None:
            if access & 2 and row['owner'] != self.owner:
                return False
            if not access & 7 and not same_package:
                return False
            access_owner = (row.get('enclosing_owner') or row['owner']) if row['kind'] == 'CLASS' else declaring
            if access & 4 and not same_package and (constructor or not self._subtype(self.owner, access_owner)):
                return False
        elif row.get('visibility') == 'private' and row['owner'] != self.owner:
            return False
        if constructor and owner.get('access', 0) & (0x200 | 0x400):
            return False
        if row.get('kind') != 'CLASS':
            for type_ in [row.get('return_type', ''), *parameter_types(row)]:
                base = type_.split('[', 1)[0]
                if self.symbols.get(base, {}).get('kind') == 'CLASS' and not self.accessible(base):
                    return False
        return True

    def _source_targets(self, unit):
        result = set()
        for name in [unit.get('owner', ''), *unit.get('imports', ())]:
            result.update(self.aliases.get(symbol_query_key(name.removeprefix('static ')), ()))
        # Source references are relation hints only. Each linked owner must
        # already exist in this target; no old donor name becomes an API.
        for token in re.findall(r'[A-Za-z_$][A-Za-z0-9_.$]*', unit.get('code', '')):
            matches = self.aliases.get(symbol_query_key(token), ())
            if len(matches) == 1:
                result.update(matches)
        return result

    def source_goal(self, purpose):
        """Register a source selector without searching the target API inventory."""
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOAL_INVALID')
        return self._goal({'purpose': purpose.strip(),
                           'kinds': ['CLASS', 'CONSTRUCTOR', 'FIELD', 'METHOD'],
                           'scope': 'package:'})

    def _rank_sources(self, purpose):
        query = _words(purpose)
        if query == self.source_ranking_query:
            return self.source_ranking
        vector = None
        if self.dense and self.source_chunks:
            if query not in self.query_vectors:
                rows = self.router.embed([query])
                if len(rows) != 1 or not rows[0]:
                    raise ValueError('CONSTRUCTION_RETRIEVAL_EMBEDDING_SHAPE')
                self.query_vectors[query] = rows[0]
            vector = self.query_vectors[query]
        sources = rank_context_chunks(query, self.source_chunks, query_vector=vector,
                                      prepared_text=self.source_prepared)
        self.source_ranking_query = query
        self.source_ranking = (sources, vector)
        return self.source_ranking

    def _rank_method_intent(self, query, kinds, eligible, relation):
        self.name_weights = _purpose_name_weights(query)
        query_terms = set(_meaningful_terms(query))
        self.strong_name_prefixes = frozenset(prefix for prefix, weight in self.name_weights.items()
                                              if weight > 1 and prefix in self.member_name_owners)
        self.exact_action_available = False
        if 'METHOD' in kinds:
            focus = self.strong_name_prefixes or self.name_weights.keys()
            matched_owners = set()
            for prefix in focus:
                matched_owners.update(self.member_name_owners.get(prefix, ()))
            for owner in matched_owners:
                keys = eligible.get(owner, ())
                if not keys:
                    continue
                best = 0.0
                for key in keys:
                    row = self.symbols[key]
                    if row['kind'] != 'METHOD':
                        continue
                    score = self._member_name_score(key)
                    best = max(best, score)
                    parts = _name_parts(row)
                    if (self.strong_name_prefixes and len(parts) == 1
                            and parts[0][:5] in self.strong_name_prefixes):
                        self.exact_action_available = True
                if best:
                    # A real matching callable outranks a class whose *name*
                    # merely resembles the requested operation. Name precision
                    # discourages registerDebugValues for "registration".
                    relation[owner] += ((8.0 if self.strong_name_prefixes else 2.0) * best
                                        + 14.0 * min(best, 1.0) * _owner_name_precision(owner, query_terms))

    def prepare(self, purpose, *, kinds, selection=None):
        if (not isinstance(purpose, str) or not purpose.strip() or not isinstance(kinds, (list, tuple, set, frozenset))
                or not kinds or any(not isinstance(kind, str) for kind in kinds)
                or not set(kinds) <= {'CLASS', 'METHOD', 'CONSTRUCTOR', 'FIELD'}):
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOAL_INVALID')
        descriptor = {'purpose': purpose.strip(), 'kinds': sorted(set(kinds)), 'scope': 'package:'}
        if selection is not None:
            descriptor['selection'] = _selection_contract(selection)
        goal = self._goal(descriptor)
        if goal == self.ordering_goal:
            return goal
        query = _words(descriptor['purpose'])
        sources, vector = self._rank_sources(descriptor['purpose'])
        relation = defaultdict(float)
        for hit in sources:
            if hit.score <= 0:
                continue
            for owner in self.source_targets[hit.chunk_id]:
                # Repeated imports are not independent behavioral evidence.
                relation[owner] = max(relation[owner], hit.score)
        for hint in self.contract.get('host_requirements', {}).get('symbols', ()):
            for owner in self.aliases.get(symbol_query_key(hint), ()):
                relation[owner] += 1.0
        for type_ in re.findall(r'[A-Za-z_$][A-Za-z0-9_.$]*', purpose):
            for owner in self.aliases.get(symbol_query_key(type_), ()):
                relation[owner] += 1.0
        # Input/output relationships expose adjacent APIs even when their
        # identifiers do not overlap the query (builders, callbacks, holders).
        for owner, strength in list(relation.items()):
            for key in self.by_owner.get(owner, ()):
                row = self.symbols[key]
                for type_ in [row.get('return_type', ''), *parameter_types(row),
                              row.get('superclass', ''), *row.get('interfaces', ())]:
                    for linked in self.aliases.get(symbol_query_key(type_.removesuffix('[]')), ()):
                        if linked != owner:
                            relation[linked] = max(relation[linked], strength / 2)
        self.anchors = {owner for owner, score in relation.items() if score > 0}
        terms = set(_meaningful_terms(query))
        # Vocabulary spread over thousands of unrelated declarations is not
        # a matching declaration. Normalize term evidence per actual member
        # document; never join an owner's entire method/constant dictionary.
        for owner, counts in self.owner_term_counts.items():
            documents = sum(self.owner_document_counts[owner][kind] for kind in kinds)
            if documents and terms:
                hits = sum(counts[kind][term] for kind in kinds for term in terms)
                relation[owner] += hits / (documents * len(terms))
        hinted_prefix = _explicit_method_prefix(descriptor['purpose']) if 'METHOD' in kinds else None
        eligible = self._focused_method_owners(
            hinted_prefix, descriptor['purpose'], descriptor.get('selection'))
        # When the current operation explicitly names a callable, perform a
        # bounded member-name lookup before owner ranking. This is an inverted
        # posting lookup, not a scan or pagination over 300k unrelated methods.
        # If the suggested callable is nonexistent, keep the ordinary discovery
        # path instead of treating the suggestion as actual API authority.
        focused = bool(eligible)
        if not eligible:
            eligible = self._eligible_owners(kinds, descriptor.get('selection'))
        self._rank_method_intent(query, kinds, eligible, relation)
        if focused:
            self.name_weights[hinted_prefix] = max(8, self.name_weights.get(hinted_prefix, 0))
            self.strong_name_prefixes = frozenset((*self.strong_name_prefixes, hinted_prefix))
            self.exact_action_available = True
        owners = rank_context_chunks(query, self.owner_chunks.values(), relation_scores=relation,
                                     prepared_text=self.owner_prepared)
        ranked = tuple(row.chunk_id for row in owners if row.chunk_id in eligible)
        self.owner_scores = {row.chunk_id: row.score for row in owners}
        self.scope_counts = defaultdict(int)
        self.scope_scores = defaultdict(float)
        for owner, keys in eligible.items():
            for scope in self.owner_scopes[owner]:
                self.scope_counts[scope] += len(keys)
                self.scope_scores[scope] = max(self.scope_scores[scope], self.owner_scores.get(owner, 0.0))
        self.query_terms, self.query_lower = set(_meaningful_terms(query)), query.casefold()
        self.member_orderings = {}
        self.scoped_orderings = {}
        self.ordering_goal = goal
        self.active_eligible = eligible
        self.ordering = (ranked, tuple(row.chunk_id for row in sources))
        print('CONSTRUCTION RETRIEVAL: ' + json.dumps({
            'goal_id': goal, 'purpose': descriptor['purpose'], 'symbol_kinds': descriptor['kinds'],
            'candidate_count': sum(len(keys) for keys in eligible.values()), 'source_unit_count': len(sources),
            'owner_count': len(ranked), 'retrieval_surface': 'inspected_target_classpath',
            'discovery_mode': 'explicit_method_postings' if focused else 'owner_representatives_then_package_type_scopes',
            'source_embeddings_used': bool(vector), 'semantic_selector': 'native_package_type_signature_selection',
        }, ensure_ascii=False), flush=True)
        return goal

    def page(self, goal, offset):
        descriptor = self._ensure(goal)
        eligible = self.active_eligible
        scope = descriptor['scope']
        representatives = scope.startswith('package:')
        if scope not in self.scoped_orderings:
            owners = tuple(owner for owner in self.ordering[0] if owner in self.scope_owners[scope])
            cumulative, total = [], 0
            for owner in owners:
                total += len(eligible[owner])
                cumulative.append(total)
            self.scoped_orderings[scope] = (owners, tuple(cumulative), len(owners) if representatives else total)
        owners, cumulative, total = self.scoped_orderings[scope]
        if type(offset) is not int or offset < 0 or (offset and offset >= total):
            raise ValueError('CONSTRUCTION_LOOKUP_OFFSET_INVALID')
        selected, cursor = [], offset

        def members(owner):
            if owner not in self.member_orderings:
                self.member_orderings[owner] = tuple(sorted(eligible[owner], key=lambda key: (
                    -self._member_name_score(key), -self._member_lexical_score(key), key)))
            return self.member_orderings[owner]

        if representatives:
            # One owner's thousands of members must not hide every other API.
            # These are discovery suggestions; full signatures/overloads stay
            # reachable through the returned verified type scopes.
            selected = [members(owner)[0] for owner in owners[offset:offset + PAGE_SIZE]]
            cursor += len(selected)
        owner_index = bisect_right(cumulative, offset)
        # Rank members only for owners on this actual page. A new purpose
        # never tokenizes or sorts the complete method/field inventory.
        while not representatives and owner_index < len(owners) and len(selected) < PAGE_SIZE:
            owner = owners[owner_index]
            start = cursor - (cumulative[owner_index - 1] if owner_index else 0)
            page = members(owner)[start:start + PAGE_SIZE - len(selected)]
            selected.extend(page)
            cursor += len(page)
            owner_index += 1
        result = {'matches': [symbol_record(key, self.symbols[key]) for key in selected],
                  'offset': offset, 'total_count': total, 'status': 'matches' if selected else 'no_matches',
                  'next_offset': cursor if cursor < total else None}
        owner_scopes = []
        for owner in dict.fromkeys(self.symbols[key]['owner'] for key in selected):
            owner_scopes.append({'goal_id': self.scoped_goal(goal, 'type:' + owner),
                                 'scope_id': 'type:' + owner, 'kind': 'type', 'name': owner,
                                 'candidate_count': len(eligible[owner])})
        print('CONSTRUCTION LOOKUP: ' + json.dumps({
            'goal_id': goal, 'purpose': descriptor['purpose'], 'scope': scope,
            'offset': offset, 'page_total': total, 'next_offset': result['next_offset'],
            'mode': 'owner_representatives' if representatives else 'type_members',
            'returned_symbols': selected,
        }, ensure_ascii=False), flush=True)
        return {'kind': 'symbols', 'query': goal, **result,
                'purpose': descriptor['purpose'], 'symbol_kinds': descriptor['kinds'], 'scope': scope,
                'owner_scopes': owner_scopes,
                'source_candidates': [self.source_record(key) for key in self.ordering[1][:PAGE_SIZE]]}

    def scope_page(self, goal, offset):
        descriptor = self._ensure(goal)
        scope = descriptor['scope']
        candidates = set(self.scope_children.get(scope, ()))
        if scope == 'package:':
            candidates.update('type:' + owner for owner in self.anchors)
        if scope.startswith('type:'):
            owner = scope.removeprefix('type:')
            for key in self.by_owner.get(owner, ()):
                row = self.symbols[key]
                for type_ in [row.get('return_type', ''), *parameter_types(row),
                              row.get('superclass', ''), *row.get('interfaces', ())]:
                    for linked in self.aliases.get(symbol_query_key(type_.split('[', 1)[0]), ()):
                        if linked != owner:
                            candidates.add('type:' + linked)
        # Skip only actual unary package chains. No branch/type is removed;
        # the returned ID still names a verified scope in this same inventory.
        compacted = set()
        for key in candidates:
            children = [child for child in self.scope_children.get(key, ()) if self.scope_counts.get(child)]
            while key.startswith('package:') and len(children) == 1 and children[0].startswith('package:'):
                key = children[0]
                children = [child for child in self.scope_children.get(key, ()) if self.scope_counts.get(child)]
            if self.scope_counts.get(key):
                compacted.add(key)
        candidates = sorted(compacted,
                            key=lambda key: (-self.scope_scores[key], not key.startswith('type:'), key))
        result = page_result(candidates, offset, field='scopes')
        rows = []
        for key in result['scopes']:
            kind, name = key.split(':', 1)
            child_goal = self._goal({**descriptor, 'scope': key})
            row = {'goal_id': child_goal, 'scope_id': key, 'kind': kind, 'name': name,
                   'candidate_count': self.scope_counts[key]}
            if kind == 'type' and name in self.symbols:
                row['declaration'] = symbol_record(name, self.symbols[name])
            rows.append(row)
        result['scopes'] = rows
        return {**result, 'scope': scope, 'purpose': descriptor['purpose']}

    def source_record(self, key):
        ref, offset, unit = self.sources[key]
        return {'source_ref': ref, 'offset': offset, 'signature': unit.get('signature', ''),
                'authority': unit.get('authority', 'inspected_materialized_source'),
                'source_sha256': unit.get('source_sha256', '')}

    def source_page(self, goal, offset):
        # Source discovery follows the same purpose-ranked complete unit map;
        # the caller selects returned pointers, never guesses a filename.
        if goal not in self.goals:
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOAL_MISSING')
        sources, _vector = self._rank_sources(self.goals[goal]['purpose'])
        result = page_result(tuple(row.chunk_id for row in sources), offset, field='sources')
        result['sources'] = [self.source_record(key) for key in result['sources']]
        return result

    def export_goals(self):
        return [{'goal_id': key, **row} for key, row in sorted(self.goals.items())]

    def restore_goals(self, rows):
        if not isinstance(rows, list):
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
        seen = set()
        for row in rows:
            if (not isinstance(row, dict) or not {'goal_id', 'purpose', 'kinds', 'scope'} <= set(row)
                    or set(row) - {'goal_id', 'purpose', 'kinds', 'scope', 'selection'}
                    or not isinstance(row['goal_id'], str) or row['goal_id'] in seen
                    or not isinstance(row['purpose'], str) or not row['purpose'].strip()
                    or not isinstance(row['kinds'], list) or not row['kinds']
                    or any(not isinstance(kind, str) for kind in row['kinds'])
                    or not isinstance(row['scope'], str) or row['scope'] not in self.scope_owners):
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            descriptor = {'purpose': row['purpose'].strip(), 'kinds': sorted(set(row['kinds'])), 'scope': row['scope']}
            if 'selection' in row:
                descriptor['selection'] = _selection_contract(row['selection'])
            if not descriptor['purpose'] or not set(descriptor['kinds']) <= {'CLASS', 'METHOD', 'CONSTRUCTOR', 'FIELD'}:
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            goal = 'intent:' + hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:24]
            if goal != row['goal_id']:
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            self.goals[goal] = descriptor
            seen.add(goal)
