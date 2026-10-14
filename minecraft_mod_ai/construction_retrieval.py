"""Purpose-driven discovery of real target APIs and complete source evidence.

Names are returned evidence, never a prerequisite invented by the author.
The target inventory remains authoritative; old source is porting evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from bisect import bisect_right
from dataclasses import replace

from .construction_state import PAGE_SIZE, page_result, symbol_query_key, symbol_record
from .java_operation_ir import parameter_types, PRIMITIVES, WIDENING, BOXING
from .rag_index import RAGChunk, rank_context_chunks, _lexical_score, _meaningful_terms, _tokens


def _words(text):
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(text))
    return re.sub(r'[.$#/_:<>(),;\[\]-]+', ' ', text)


def _description(row):
    return _words(' '.join(str(value) for value in (
        row.get('owner', ''), row.get('name', ''), row.get('kind', ''),
        row.get('purpose', ''), row.get('return_type', ''),
        *parameter_types(row), row.get('superclass', ''), *row.get('interfaces', ()),
        *row.get('functional_parameters', ()), row.get('functional_return_type', ''),
    )))


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
        self.by_owner = defaultdict(list)
        self.aliases = defaultdict(set)
        self.member_text = {}
        self.member_prepared = {}
        self.scope_children = defaultdict(set)
        self.scope_owners = defaultdict(set)
        self.owner_scopes = {}
        self.eligible = {}
        self.eligibility_header = None
        self.member_orderings = {}
        descriptions = defaultdict(list)
        for key, row in symbols.items():
            if row.get('provenance') == 'construction_plan' and row['owner'] != owner:
                continue
            self.by_owner[row['owner']].append(key)
            descriptions[row['owner']].append(_description(row))
            if row.get('kind') == 'CLASS':
                self.aliases[symbol_query_key(key)].add(key)
                self.aliases[symbol_query_key(row['name'])].add(key)
        self.owner_chunks = tuple(RAGChunk(owner, owner, '\n'.join(parts), 1, 1, '', {})
                                  for owner, parts in sorted(descriptions.items()))
        self.owner_prepared = {row.chunk_id: (frozenset(_tokens(row.text)), row.text.casefold())
                               for row in self.owner_chunks}
        for type_owner in self.by_owner:
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

    def _add_owner_scope(self, owner):
        package = owner.rpartition('.')[0]
        scopes = ['package:']
        parts = package.split('.') if package else []
        for index in range(1, len(parts) + 1):
            scope = 'package:' + '.'.join(parts[:index])
            self.scope_children[scopes[-1]].add(scope)
            scopes.append(scope)
        leaf = 'type:' + owner
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
        self.member_text[key] = _description(row)
        self.member_prepared[key] = (frozenset(_tokens(self.member_text[key])), self.member_text[key].casefold())
        self.eligible.clear()
        self.ordering_goal = None

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
            self.prepare(row['purpose'], kinds=row['kinds'])
        return row

    def root_goal(self, goal):
        return self._goal({**self.goals[goal], 'scope': 'package:'})

    def _eligible_owners(self, kinds):
        header = self.symbols.get(self.owner, {})
        fingerprint = (header.get('superclass', ''), tuple(header.get('interfaces', ())))
        if fingerprint != self.eligibility_header:
            self.eligible.clear()
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
        return self.eligible[kinds]

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
                pending.extend(self.aliases.get(symbol_query_key(parent), ()))
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
        same_package = row['owner'].rpartition('.')[0] == self.owner.rpartition('.')[0]
        if owner and owner.get('side') == 'CLIENT' and not self.target_path.startswith('src/client/'):
            return False
        if owner.get('access', 1) & 2 and row['owner'] != self.owner:
            return False
        if not owner.get('access', 1) & 7 and not same_package:
            return False
        access = row.get('access')
        if access is not None:
            if access & 2 and row['owner'] != self.owner:
                return False
            if not access & 7 and not same_package:
                return False
            if access & 4 and not same_package and (constructor or not self._subtype(self.owner, row['owner'])):
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

    def prepare(self, purpose, *, kinds):
        if (not isinstance(purpose, str) or not purpose.strip() or not isinstance(kinds, (list, tuple, set, frozenset))
                or not kinds or any(not isinstance(kind, str) for kind in kinds)
                or not set(kinds) <= {'CLASS', 'METHOD', 'CONSTRUCTOR', 'FIELD'}):
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOAL_INVALID')
        descriptor = {'purpose': purpose.strip(), 'kinds': sorted(set(kinds)), 'scope': 'package:'}
        goal = self._goal(descriptor)
        if goal == self.ordering_goal:
            return goal
        query = _words(descriptor['purpose'])
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
        owners = rank_context_chunks(query, self.owner_chunks, relation_scores=relation,
                                     prepared_text=self.owner_prepared)
        eligible = self._eligible_owners(kinds)
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
        self.ordering = (ranked, tuple(row.chunk_id for row in sources))
        print('CONSTRUCTION RETRIEVAL: ' + json.dumps({
            'goal_id': goal, 'purpose': descriptor['purpose'], 'symbol_kinds': descriptor['kinds'],
            'candidate_count': sum(len(keys) for keys in eligible.values()), 'source_unit_count': len(sources),
            'source_embeddings_used': bool(vector), 'semantic_selector': 'native_package_type_signature_selection',
        }, ensure_ascii=False), flush=True)
        return goal

    def page(self, goal, offset):
        descriptor = self._ensure(goal)
        eligible = self._eligible_owners(descriptor['kinds'])
        scope = descriptor['scope']
        if scope not in self.scoped_orderings:
            owners = tuple(owner for owner in self.ordering[0] if owner in self.scope_owners[scope])
            cumulative, total = [], 0
            for owner in owners:
                total += len(eligible[owner])
                cumulative.append(total)
            self.scoped_orderings[scope] = (owners, tuple(cumulative), total)
        owners, cumulative, total = self.scoped_orderings[scope]
        if type(offset) is not int or offset < 0 or (offset and offset >= total):
            raise ValueError('CONSTRUCTION_LOOKUP_OFFSET_INVALID')
        selected, cursor = [], offset
        owner_index = bisect_right(cumulative, offset)
        # Rank members only for owners on this actual page. A new purpose
        # never tokenizes or sorts the complete method/field inventory.
        while owner_index < len(owners) and len(selected) < PAGE_SIZE:
            owner = owners[owner_index]
            if owner not in self.member_orderings:
                self.member_orderings[owner] = tuple(sorted(eligible[owner], key=lambda key: (
                    -self._member_lexical_score(key), key)))
            start = cursor - (cumulative[owner_index - 1] if owner_index else 0)
            page = self.member_orderings[owner][start:start + PAGE_SIZE - len(selected)]
            selected.extend(page)
            cursor += len(page)
            owner_index += 1
        result = {'matches': [symbol_record(key, self.symbols[key]) for key in selected],
                  'offset': offset, 'total_count': total, 'status': 'matches' if selected else 'no_matches',
                  'next_offset': cursor if cursor < total else None}
        return {'kind': 'symbols', 'query': goal, **result,
                'purpose': descriptor['purpose'], 'symbol_kinds': descriptor['kinds'], 'scope': scope,
                'source_candidates': [self.source_record(key) for key in self.ordering[1][:PAGE_SIZE]]}

    def scope_page(self, goal, offset):
        descriptor = self._ensure(goal)
        scope = descriptor['scope']
        candidates = set(self.scope_children.get(scope, ()))
        if scope.startswith('type:'):
            owner = scope.removeprefix('type:')
            for key in self.by_owner.get(owner, ()):
                row = self.symbols[key]
                for type_ in [row.get('return_type', ''), *parameter_types(row),
                              row.get('superclass', ''), *row.get('interfaces', ())]:
                    for linked in self.aliases.get(symbol_query_key(type_.split('[', 1)[0]), ()):
                        if linked != owner:
                            candidates.add('type:' + linked)
        candidates = sorted((key for key in candidates if self.scope_counts.get(key)),
                            key=lambda key: (-self.scope_scores[key], key))
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
        self._ensure(goal)
        result = page_result(self.ordering[1], offset, field='sources')
        result['sources'] = [self.source_record(key) for key in result['sources']]
        return result

    def export_goals(self):
        return [{'goal_id': key, **row} for key, row in sorted(self.goals.items())]

    def restore_goals(self, rows):
        if not isinstance(rows, list):
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
        seen = set()
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {'goal_id', 'purpose', 'kinds', 'scope'}
                    or not isinstance(row['goal_id'], str) or row['goal_id'] in seen
                    or not isinstance(row['purpose'], str) or not row['purpose'].strip()
                    or not isinstance(row['kinds'], list) or not row['kinds']
                    or any(not isinstance(kind, str) for kind in row['kinds'])
                    or not isinstance(row['scope'], str) or row['scope'] not in self.scope_owners):
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            descriptor = {'purpose': row['purpose'].strip(), 'kinds': sorted(set(row['kinds'])), 'scope': row['scope']}
            if not descriptor['purpose'] or not set(descriptor['kinds']) <= {'CLASS', 'METHOD', 'CONSTRUCTOR', 'FIELD'}:
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            goal = 'intent:' + hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:24]
            if goal != row['goal_id']:
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            self.goals[goal] = descriptor
            seen.add(goal)
