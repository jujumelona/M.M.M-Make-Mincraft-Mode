"""Purpose-driven discovery of real target APIs and complete source evidence.

Names are returned evidence, never a prerequisite invented by the author.
The target inventory remains authoritative; old source is porting evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import replace

from .construction_state import PAGE_SIZE, page_result, symbol_query_key, symbol_record
from .java_operation_ir import parameter_types
from .rag_index import RAGChunk, rank_context_chunks, _lexical_score, _meaningful_terms


def _words(text):
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(text))
    return re.sub(r'[.$#/_:<>(),;\[\]-]+', ' ', text)


def _description(row):
    return _words(' '.join(str(value) for value in (
        row.get('owner', ''), row.get('name', ''), row.get('kind', ''),
        row.get('purpose', ''), row.get('return_type', ''),
        *parameter_types(row), row.get('superclass', ''), *row.get('interfaces', ()),
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
        descriptions = defaultdict(list)
        for key, row in symbols.items():
            if row.get('provenance') == 'construction_plan':
                continue
            self.by_owner[row['owner']].append(key)
            descriptions[row['owner']].append(_description(row))
            if row.get('kind') == 'CLASS':
                self.aliases[symbol_query_key(key)].add(key)
                self.aliases[symbol_query_key(row['name'])].add(key)
        self.owner_chunks = tuple(RAGChunk(owner, owner, '\n'.join(parts), 1, 1, '', {})
                                  for owner, parts in sorted(descriptions.items()))
        self.sources = {}
        for source_ref, units in source_pages.items():
            for offset, unit in enumerate(units):
                if unit.get('kind') in {'class', 'interface', 'enum', 'record'}:
                    # Complete member units carry the same code without adding
                    # a duplicate full compilation unit to every context.
                    continue
                if unit.get('side') == 'CLIENT' and not target_path.startswith('src/client/'):
                    continue
                key = source_ref + ':' + str(offset)
                self.sources[key] = (source_ref, offset, unit)
        self.source_chunks = tuple(
            RAGChunk(key, ref, _words(unit.get('signature', '') + '\n' + unit.get('code', '')),
                     unit.get('start_line', 1), unit.get('end_line', 1),
                     unit.get('source_sha256', ''), {})
            for key, (ref, _offset, unit) in self.sources.items())
        self.source_targets = {key: self._source_targets(unit)
                               for key, (_ref, _offset, unit) in self.sources.items()}
        self.dense = self._dense_allowed()
        if self.dense and self.source_chunks:
            vectors = router.embed([row.text for row in self.source_chunks])
            if len(vectors) != len(self.source_chunks) or any(not vector for vector in vectors):
                raise ValueError('CONSTRUCTION_RETRIEVAL_EMBEDDING_SHAPE')
            self.source_chunks = tuple(replace(row, embedding=tuple(vector))
                                       for row, vector in zip(self.source_chunks, vectors))
        self.query_vectors = {}

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
        if not isinstance(purpose, str) or not purpose.strip() or not set(kinds) <= {'CLASS', 'METHOD', 'CONSTRUCTOR', 'FIELD'}:
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOAL_INVALID')
        descriptor = {'purpose': purpose.strip(), 'kinds': sorted(set(kinds))}
        goal = 'intent:' + hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:24]
        self.goals[goal] = descriptor
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
        sources = rank_context_chunks(query, self.source_chunks, query_vector=vector)
        relation = defaultdict(float)
        for hit in sources:
            if hit.score <= 0:
                continue
            for owner in self.source_targets[hit.chunk_id]:
                relation[owner] += hit.score
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
        owners = rank_context_chunks(query, self.owner_chunks, relation_scores=relation)
        owner_scores = {row.chunk_id: row.score for row in owners}
        # New own declarations are valid local targets, distinct from donor or
        # installed API authority. Their actual purpose remains visible.
        terms = set(_meaningful_terms(query))
        ranked = []
        for key, row in self.symbols.items():
            if row.get('kind') not in kinds or not self.accessible(key, constructor=row.get('kind') == 'CONSTRUCTOR'):
                continue
            text = _description(row)
            score = owner_scores.get(row['owner'], 0.0) + _lexical_score(terms, query.casefold(), text)
            ranked.append((score, key))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        self.ordering_goal = goal
        self.ordering = (tuple(key for _score, key in ranked), tuple(row.chunk_id for row in sources))
        print('CONSTRUCTION RETRIEVAL: ' + json.dumps({
            'goal_id': goal, 'purpose': descriptor['purpose'], 'symbol_kinds': descriptor['kinds'],
            'candidate_count': len(ranked), 'source_unit_count': len(sources),
            'source_embeddings_used': bool(vector), 'semantic_selector': 'native_real_candidate_selection',
        }, ensure_ascii=False), flush=True)
        return goal

    def page(self, goal, offset):
        if goal not in self.goals:
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOAL_MISSING')
        if goal != self.ordering_goal:
            row = self.goals[goal]
            self.prepare(row['purpose'], kinds=row['kinds'])
        keys, sources = self.ordering
        result = page_result(keys, offset, field='matches')
        result['matches'] = [symbol_record(key, self.symbols[key]) for key in result['matches']]
        return {'kind': 'symbols', 'query': goal, **result,
                'purpose': self.goals[goal]['purpose'], 'symbol_kinds': self.goals[goal]['kinds'],
                'source_candidates': [self.source_record(key) for key in sources[:PAGE_SIZE]]}

    def source_record(self, key):
        ref, offset, unit = self.sources[key]
        return {'source_ref': ref, 'offset': offset, 'signature': unit.get('signature', ''),
                'authority': unit.get('authority', 'inspected_materialized_source'),
                'source_sha256': unit.get('source_sha256', '')}

    def export_goals(self):
        return [{'goal_id': key, **row} for key, row in sorted(self.goals.items())]

    def restore_goals(self, rows):
        if not isinstance(rows, list):
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'goal_id', 'purpose', 'kinds'} or row['goal_id'] in seen:
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            descriptor = {'purpose': str(row['purpose']).strip(), 'kinds': sorted(set(row['kinds']))}
            if not descriptor['purpose'] or not set(descriptor['kinds']) <= {'CLASS', 'METHOD', 'CONSTRUCTOR', 'FIELD'}:
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            goal = 'intent:' + hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:24]
            if goal != row['goal_id']:
                raise ValueError('CONSTRUCTION_RETRIEVAL_GOALS_INVALID')
            self.goals[goal] = descriptor
            seen.add(goal)
