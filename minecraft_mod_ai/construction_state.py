"""Bounded, host-owned state shared by declaration and operation authors.

Planned declarations are usable within their owner, but are not evidence of
already installed bytecode. The complete inventory and accepted plan stay on
the host; every decision receives its phase, cursor and lookup provenance.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy


# Body lookup choices and retained API pages changed. Old journals/cursors may
# not be replayed against them; declaration-bound accepted graphs stay reusable.
PROTOCOL_VERSION = "mmm/construction-state-v15"
PAGE_SIZE = 8
RECENT_SIZE = 4
LOOKUP_KINDS = frozenset({'symbols', 'scopes', 'declarations', 'nodes', 'node_index', 'source', 'sources', 'bindings'})
TYPE_LOOKUP_PHASES = frozenset({'class_header', 'member_return_type', 'member_parameter_type'})
RESELECTABLE_OPERATIONS = frozenset({'call', 'new', 'field', 'lambda'})


def symbol_query_key(query):
    # Bytecode uses '$' for nested owners; inspected source/type names use '.'.
    # They are one selector, including for consumed-query bookkeeping.
    return query.strip().casefold().replace('$', '.')


def _lookup_request(kind, query, offset):
    if kind not in LOOKUP_KINDS or not isinstance(query, str) or type(offset) is not int or offset < 0:
        raise ValueError('CONSTRUCTION_LOOKUP_REQUEST_INVALID')
    normalized = (symbol_query_key(query) if kind == 'symbols' else
                  query.strip().casefold() if kind == 'sources' else query.strip())
    if kind == 'nodes':
        normalized = ','.join(sorted(key.strip() for key in query.split(',')))
    if kind == 'declarations':
        normalized = ''
    return kind, normalized, offset


def symbol_record(key, row):
    visible = {k: v for k, v in row.items()
               if k not in {"code", "source", "source_path"}}
    provenance = row.get("provenance")
    origin = ("planned_declaration" if provenance == "construction_plan" else
              "materialized_source" if provenance == "materialized_source" else
              "inspected_bytecode")
    return {**visible, "symbol_id": key, "origin": origin}


def page_result(rows, offset, *, field):
    if type(offset) is not int or offset < 0 or (offset and offset >= len(rows)):
        raise ValueError("CONSTRUCTION_LOOKUP_OFFSET_INVALID")
    selected = rows[offset:offset + PAGE_SIZE]
    return {field: selected, "offset": offset, "total_count": len(rows),
            "status": "matches" if selected else "no_matches",
            "next_offset": offset + PAGE_SIZE if len(rows) > offset + PAGE_SIZE else None}


def symbol_page(symbols, query, offset, *, kinds=None):
    query = query.strip()
    if len(query) < 2:
        raise ValueError("CONSTRUCTION_LOOKUP_QUERY_REQUIRED")
    folded = symbol_query_key(query)

    def rank(key):
        row = symbols[key]
        owner = symbol_query_key(str(row.get('owner', '')))
        name = symbol_query_key(str(row.get('name', '')))
        simple_owner = owner.rsplit('.', 1)[-1].rsplit('$', 1)[-1]
        if symbol_query_key(key) == folded or (row.get('kind') == 'CLASS'
                and (folded in {name, simple_owner} or owner.endswith('.' + folded))):
            return 0, key
        if name == folded:
            return 1, key
        if folded in {owner, simple_owner}:
            return 2, key
        return 3, key

    # The first page must expose an exact type before unrelated fields or
    # method descriptors that merely mention it. Keep every remaining result
    # and a stable order so subsequent pages still exhaust the same inventory.
    matches = sorted((key for key, row in symbols.items()
                      if row.get('provenance') != 'construction_plan'
                      and (kinds is None or row.get('kind') in kinds)
                      and (folded in symbol_query_key(key)
                           or folded == symbol_query_key(str(row.get('name', ''))))),
                     key=rank)
    # Materialize only this page, never the whole inspected inventory.
    result = page_result(matches, offset, field="matches")
    result["matches"] = [symbol_record(key, symbols[key]) for key in result["matches"]]
    return {"kind": "symbols", "query": query, **result,
            **({'symbol_kinds': sorted(kinds)} if kinds is not None else {})}


STATE_INSTRUCTION = (
    "construction_state is the host's current accepted plan. A new owner and its "
    "planned_declaration symbols are being created, not missing installed source. "
    "Its class header and recent declarations are already shown; use declarations "
    "to page older accepted members when that action is available. Do not search "
    "for an existing implementation of a new owner. An empty lookup means no match "
    "for that query, not missing context. A lookup's next_offset=null means its "
    "results are exhausted. Never repeat a consumed query and offset in the same "
    "phase and revision; choose a returned scope/successor or advance the current "
    "declaration/operation. Planned members with a body_owner are already assigned; "
    "add only missing declarations. authoring_contract defines the artifact's responsibility, "
    "identities and implementation requirements. Use bindings with an exact binding_id from "
    "authoring_contract.binding_pages to inspect shared configuration before choosing declarations "
    "or types. evidence_memory.working_set retains exact inspected facts across phases. "
    "Use memory_index to page its complete catalogue and recall an exact evidence_id when an "
    "earlier fact is outside the active view. Recall is memory navigation, not a new search. "
    "Revisiting a memory catalogue page requires an actual fact recall before more catalogue "
    "navigation. Pages containing only active facts add no information. Exact older facts "
    "remain recoverable; recalling one is not declaring new work. "
    "Dependency discovery is driven by the current implementation purpose. Exact API names come "
    "from returned candidates, not guesses. Intent selectors identify host-ranked real pages. "
    "Package symbol pages show representative signatures from different owners; they are not "
    "an exhaustive member scan. Select a returned package/type scope for a specific missing API. "
    "A type scope retains its complete overload/member pages. Choose a Java operation before "
    "resolving its API; intrinsic/local operations do not need a dependency search. "
    "scope_candidates are real package/type branches, including branches whose names do not "
    "overlap the purpose. Select their literal goal_id to narrow by meaning and inspect actual signatures. "
    "Select only available_actions."
)


class ConstructionState:
    def __init__(self, symbols, owner, target_path, *, declaration_ids=(), revision=0,
                 authoring_contract=None, binding_pages=None, evidence_index=None):
        self.symbols = symbols
        self.owner = owner
        self.target_path = target_path
        self.authoring_contract = dict(authoring_contract or {})
        self.binding_pages = dict(binding_pages or {})
        self.declaration_ids = list(declaration_ids)
        self.revision = revision
        self.phase = ""
        self.lookup = {}
        self.symbol_lookup = {}
        self.binding_lookup = {}
        self.binding_lookups = {}
        self.source_lookup = {}
        self.scope_lookup = {}
        self.history = deque(maxlen=PAGE_SIZE)
        self.requests = set()
        self.lookup_cursors = {}
        self.evidence_index = evidence_index
        self.symbol_goal = None
        self.operation_goal = None
        self.operation_draft = {}
        self.selection_goal = None
        self.operation_reads = 0
        self.operation_rejections = {}
        # API-level rejections survive local revisions and checkpoint resumes.
        # Never confuse one bad symbol with the entire 'call' operation.
        self.symbol_rejections = {}
        self.node_purposes = {}
        self.validation_observation = None
        from .construction_memory import ConstructionMemory
        self.memory = ConstructionMemory()

    def enter(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.requests.clear()
            self.lookup_cursors.clear()
            self.memory.reset_navigation()
            self.operation_reads = 0

    def advance(self, *, preserve_symbols=False):
        if self.phase == 'class_header' and self.evidence_index is not None:
            self.evidence_index.include_symbol(self.owner)
        self.revision += 1
        self.lookup = {}
        if not preserve_symbols:
            self.symbol_lookup = {}
        self.scope_lookup = {}
        self.requests.clear()
        self.lookup_cursors.clear()
        self.memory.reset_navigation()
        self.symbol_goal = None
        self.operation_goal = None
        self.operation_draft = {}
        self.selection_goal = None
        self.operation_reads = 0
        self.operation_rejections = {}
        self.validation_observation = None

    def reconsider_operation(self):
        """Retry the same slot with another operation, preserving real evidence."""
        if self.phase != 'operation_graph' or self.selection_goal is None:
            raise ValueError('CONSTRUCTION_OPERATION_RECONSIDER_INVALID')
        self.lookup = {}
        self.symbol_lookup = {}
        self.scope_lookup = {}
        self.symbol_goal = None
        self.operation_goal = None
        self.operation_draft = {}
        self.requests.clear()
        self.lookup_cursors.clear()
        self.memory.reset_navigation()
        self.operation_reads = 0
        # No graph node was accepted: revision and selected slot must not change.


    def reject_operation(self, goal, reason):
        if (goal['action'] not in RESELECTABLE_OPERATIONS
                or any(not isinstance(value, str) or not value.strip() or len(value) > 240
                       for value in (goal['purpose'], reason))):
            raise ValueError('CONSTRUCTION_OPERATION_RESELECTION_INVALID')
        self.operation_rejections[goal['action']] = {'purpose': goal['purpose'], 'reason': reason}
        # This is a change of implementation strategy for the same missing
        # slot, not an accepted node or evidence that an API does not exist.
        self.operation_goal = None
        self.operation_draft = {}
        self.symbol_goal = None
        self.symbol_lookup = {}
        self.scope_lookup = {}
        if self.lookup.get('kind') in {'symbols', 'scopes'}:
            self.lookup = {}

    def reject_api(self, node_id, symbol_ref, reason):
        if (not isinstance(node_id, str) or not node_id.strip()
                or symbol_ref not in self.symbols or not isinstance(reason, str)
                or not reason.strip() or len(reason) > 240):
            raise ValueError('CONSTRUCTION_API_REJECTION_INVALID')
        self.symbol_rejections.setdefault(node_id, {})[symbol_ref] = reason.strip()

    def rejected_apis(self, node_id):
        return self.symbol_rejections.get(node_id, {})

    def remember_node_purpose(self, node_id, purpose, *, replace=False):
        if not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 240:
            raise ValueError('CONSTRUCTION_NODE_PURPOSE_INVALID')
        if replace or node_id not in self.node_purposes:
            self.node_purposes[node_id] = purpose.strip()

    def prepare_symbols(self, purpose, *, kinds=('CLASS',), preserve_scope=False):
        if self.evidence_index is None:
            return
        # Authoring rules/negative examples are constraints, not the positive
        # behavior to retrieve. They remain in authoring_contract unchanged.
        query = purpose
        goal = self.evidence_index.prepare(query, kinds=kinds)
        current = self.evidence_index.goals.get(self.symbol_goal)
        if current is None or {**current, 'scope': 'package:'} != self.evidence_index.goals[goal]:
            self.symbol_goal = (self.evidence_index.scoped_goal(goal, current['scope'])
                                if preserve_scope and current is not None else goal)
        if self.symbol_goal != goal and _lookup_request('scopes', goal, 0) not in self.requests:
            # Narrowing to a kind keeps the inspected scope, with a real root
            # frontier still available if its relevant member lives elsewhere.
            self.consume_lookup('scopes', goal, 0, self.evidence_index.scope_page(goal, 0))
        if _lookup_request('symbols', self.symbol_goal, 0) not in self.requests:
            self.lookup_symbols(self.symbol_goal, 0)
        if _lookup_request('scopes', self.symbol_goal, 0) not in self.requests:
            self.consume_lookup('scopes', self.symbol_goal, 0,
                                self.evidence_index.scope_page(self.symbol_goal, 0))

    def add_declaration(self, key):
        if key in self.declaration_ids:
            raise ValueError("CONSTRUCTION_DECLARATION_DUPLICATE: " + key)
        self.declaration_ids.append(key)
        if self.evidence_index is not None:
            self.evidence_index.include_symbol(key)
        self.advance()

    def scope_candidates(self):
        if self.evidence_index is None or self.symbol_goal is None:
            return []
        descriptor = self.evidence_index.goals[self.symbol_goal]
        result = {}
        # The currently visible representative owners are direct, verified
        # type scopes. They share the same bounded frontier as package scopes;
        # a second model-visible list must not double its cardinality.
        for candidate in self.symbol_lookup.get('owner_scopes', ()):
            goal = candidate['goal_id']
            row = self.evidence_index.goals.get(goal)
            if (row is not None
                    and {**row, 'scope': 'package:'} == {**descriptor, 'scope': 'package:'}
                    and ('scopes', goal, 0) not in self.requests):
                result[goal] = candidate
                if len(result) == PAGE_SIZE:
                    return list(result.values())
        for kind, query, offset in sorted(self.requests):
            if kind != 'scopes':
                continue
            row = self.evidence_index.goals[query]
            if {**row, 'scope': 'package:'} != {**descriptor, 'scope': 'package:'}:
                continue
            for candidate in self.evidence_index.scope_page(query, offset)['scopes']:
                if ('scopes', candidate['goal_id'], 0) not in self.requests:
                    result[candidate['goal_id']] = candidate
                    # The complete frontier is reconstructible from consumed
                    # selectors on the host. Expose one page; consuming a
                    # candidate reveals its next waiting sibling. Accumulating
                    # every visited page here overflowed the live 32K slot.
                    if len(result) == PAGE_SIZE:
                        return list(result.values())
        return list(result.values())

    def payload(self, available_actions, **cursor):
        from .construction_feedback import observation_view
        available_actions = tuple(available_actions)
        row = self.symbols.get(self.owner, {})
        planned = row.get("provenance") == "construction_plan"
        return {"construction_state": {
            "protocol": PROTOCOL_VERSION, "phase": self.phase, "revision": self.revision,
            "owner": {"symbol_id": self.owner, "target_path": self.target_path,
                      "status": "under_construction" if planned else "existing_inspected",
                      "header": symbol_record(self.owner, row)},
            "declarations": {"total_count": len(self.declaration_ids),
                "recent": [symbol_record(key, self.symbols[key])
                           for key in self.declaration_ids[-RECENT_SIZE:]]},
            "lookups_since_progress": max(len(self.requests), self.operation_reads),
            "operation_context_reads": self.operation_reads,
            "operation_rejections": deepcopy(self.operation_rejections),
            "symbol_rejections": deepcopy(self.symbol_rejections),
            "recent_lookups": list(self.history),
            "available_actions": list(available_actions), **cursor,
        }, "lookup": {key: value for key, value in self.lookup.items()
                      if key not in {'matches', 'source', 'bindings', 'declarations', 'scopes'}
                      and (key != 'nodes' or self.lookup.get('kind') == 'node_index')},
            "authoring_contract": self.authoring_contract,
            "evidence_memory": self.memory.payload(),
            "selection_goal": deepcopy(self.selection_goal),
            "operation_observation": observation_view(self.validation_observation),
            "scope_candidates": self.scope_candidates() if 'scopes' in available_actions else [],
            "scope_lookup": self.scope_lookup,
            "symbol_page_choices": self.symbol_page_successors() if 'more_symbols' in available_actions else [],
            "binding_lookup": {key: value for key, value in self.binding_lookup.items() if key != 'bindings'},
            "source_lookup": {key: value for key, value in self.source_lookup.items() if key != 'source'},
            "symbol_lookup": {key: value for key, value in self.symbol_lookup.items() if key != 'matches'}}

    def _remember_cursor(self, kind, query, offset, result):
        request = _lookup_request(kind, query, offset)
        previous = self.lookup_cursors.get(request[:2])
        if previous is not None and previous['offset'] > offset:
            # A purpose-ranked source pointer can be read out of file order.
            # The resume cursor still records the greatest consumed offset.
            return
        self.lookup_cursors[request[:2]] = {
            'kind': kind, 'query': query.strip(), 'offset': offset,
            'next_offset': result['next_offset'],
        }

    def lookup_successors(self, kind):
        pages = {key: row for key, row in self.lookup_cursors.items() if key[0] == kind}
        # A real dependency page remains usable after an accepted graph node.
        # Its exact successor is still resolved from the same target inventory.
        if kind == 'symbols' and self.symbol_lookup:
            row = self.symbol_lookup
            key = _lookup_request(kind, row['query'], row['offset'])[:2]
            pages.setdefault(key, row)
        return [row for row in pages.values() if row['next_offset'] is not None
                and _lookup_request(kind, row['query'], row['next_offset']) not in self.requests]

    def symbol_page_successors(self):
        if self.evidence_index is None or self.symbol_goal is None:
            return []
        current = self.evidence_index.goals[self.symbol_goal]
        result = []
        for row in self.lookup_successors('symbols'):
            descriptor = self.evidence_index.goals[row['query']]
            if (descriptor['scope'].startswith('type:')
                    and {**descriptor, 'scope': 'package:'} == {**current, 'scope': 'package:'}):
                result.append({'query': row['query'], 'offset': row['next_offset'],
                               'scope': descriptor['scope'], 'purpose': descriptor['purpose']})
        return result

    def consume_lookup(self, kind, query, offset, result):
        request = _lookup_request(kind, query, offset)
        if request in self.requests:
            raise ValueError("CONSTRUCTION_AUTHOR_NO_PROGRESS: repeated " + kind + " lookup")
        self.requests.add(request)
        self.lookup = {**result, "kind": kind, "query": query.strip(), "offset": offset}
        self.memory.remember(kind, query, offset, result)
        self._remember_cursor(kind, query, offset, result)
        if kind == 'symbols':
            self.symbol_lookup = self.lookup
            if self.evidence_index is not None:
                self.symbol_goal = query.strip()
        if kind == 'bindings':
            self.binding_lookup = self.lookup
            self.binding_lookups[(query.strip(), offset)] = self.lookup
        if kind == 'source':
            self.source_lookup = self.lookup
        if kind == 'scopes':
            self.scope_lookup = self.lookup
        self.history.append({"phase": self.phase, "revision": self.revision,
            "kind": kind, "query": query.strip(), "offset": offset,
            "status": result["status"], "total_count": result["total_count"],
            "next_offset": result["next_offset"]})
        return self.lookup

    def export(self):
        # Save selectors, never cached source or API rows as fresh authority.
        last = {key: self.lookup[key] for key in ('kind', 'query', 'offset')} if self.lookup else None
        binding = {key: self.binding_lookup[key] for key in ('kind', 'query', 'offset')} if self.binding_lookup else None
        symbol = {key: self.symbol_lookup[key] for key in ('kind', 'query', 'offset')} if self.symbol_lookup else None
        source = {key: self.source_lookup[key] for key in ('kind', 'query', 'offset')} if self.source_lookup else None
        scope = {key: self.scope_lookup[key] for key in ('kind', 'query', 'offset')} if self.scope_lookup else None
        return {'protocol': PROTOCOL_VERSION, 'phase': self.phase, 'revision': self.revision,
                'operation_reads': self.operation_reads,
                'requests': [list(row) for row in sorted(self.requests)],
                'history': deepcopy(list(self.history)), 'last_lookup': last,
                'last_binding_lookup': binding, 'last_symbol_lookup': symbol,
                'last_source_lookup': source,
                'last_scope_lookup': scope,
                'symbol_goal': self.symbol_goal, 'operation_goal': deepcopy(self.operation_goal),
                'operation_draft': deepcopy(self.operation_draft),
                'selection_goal': deepcopy(self.selection_goal),
                'operation_rejections': deepcopy(self.operation_rejections),
                'symbol_rejections': deepcopy(self.symbol_rejections),
                'node_purposes': dict(self.node_purposes),
                'validation_observation': deepcopy(self.validation_observation),
                'memory': self.memory.export(),
                'binding_lookups': [{'kind': 'bindings', 'query': query, 'offset': offset}
                                    for query, offset in sorted(self.binding_lookups)],
                'evidence_goals': self.evidence_index.export_goals() if self.evidence_index else [],
                'lookup_cursors': [{key: row[key] for key in ('kind', 'query', 'offset')}
                                   for _, row in sorted(self.lookup_cursors.items())]}

    def restore(self, record, *, resolve_lookup=None):
        from .java_operation_ir import OPERATIONS
        if (not isinstance(record, dict) or record.get('protocol') != PROTOCOL_VERSION
                or record.get('phase') != self.phase or type(record.get('revision')) is not int
                or record['revision'] != self.revision):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_BINDING_MISMATCH')
        if self.evidence_index is not None:
            self.evidence_index.restore_goals(record.get('evidence_goals'))
        self.symbol_goal = record.get('symbol_goal')
        if self.symbol_goal is not None and (self.evidence_index is None
                or self.symbol_goal not in self.evidence_index.goals):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        goal = record.get('operation_goal')
        if goal is not None:
            from .java_operation_ir import OPERATIONS
            import re
            if (not isinstance(goal, dict) or set(goal) != {'action', 'purpose', 'node_id'}
                    or goal['action'] not in OPERATIONS or not isinstance(goal['purpose'], str)
                    or not 1 <= len(goal['purpose']) <= 240
                    or not isinstance(goal['node_id'], str)
                    or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,240}', goal['node_id'])):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        self.operation_goal = deepcopy(goal)
        draft = record.get('operation_draft')
        if not isinstance(draft, dict) or (draft and goal is None):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        self.operation_draft = deepcopy(draft)
        selection = record.get('selection_goal')
        if selection is not None:
            import re
            if (not isinstance(selection, dict) or not {'node_id', 'purpose'} <= set(selection)
                    or set(selection) - {'node_id', 'purpose', 'excluded_operations'}
                    or not isinstance(selection.get('excluded_operations', []), list)
                    or any(not isinstance(action, str) or action not in OPERATIONS
                           for action in selection.get('excluded_operations', []))
                    or len(set(selection.get('excluded_operations', []))) != len(selection.get('excluded_operations', []))
                    or not isinstance(selection['purpose'], str) or not selection['purpose'].strip()
                    or not isinstance(selection['node_id'], str)
                    or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,240}', selection['node_id'])):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        if goal is not None and (selection is None or selection['node_id'] != goal['node_id']):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        self.selection_goal = deepcopy(selection)
        rejections = record.get('operation_rejections', {})
        purposes = record.get('node_purposes', {})
        import re
        if (not isinstance(rejections, dict) or not set(rejections) <= RESELECTABLE_OPERATIONS
                or (rejections and selection is None)
                or any(not isinstance(row, dict) or set(row) != {'purpose', 'reason'}
                       or any(not isinstance(value, str) or not value.strip() or len(value) > 240
                              for value in row.values()) for row in rejections.values())
                or (goal is not None and goal['action'] in rejections)
                or not isinstance(purposes, dict)
                or any(not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,240}', key)
                       or not isinstance(value, str) or not value.strip() or len(value) > 240
                       for key, value in purposes.items())):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        rejected_symbols = record.get('symbol_rejections', {})
        if (not isinstance(rejected_symbols, dict)
                or any(not isinstance(node_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,240}', node_id)
                       or not isinstance(rows, dict)
                       or any(ref not in self.symbols or not isinstance(reason, str)
                              or not reason.strip() or len(reason) > 240
                              for ref, reason in rows.items())
                       for node_id, rows in rejected_symbols.items())):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        self.operation_rejections = deepcopy(rejections)
        self.symbol_rejections = deepcopy(rejected_symbols)
        self.node_purposes = dict(purposes)
        from .construction_feedback import validate_observation
        observation = record.get('validation_observation')
        validate_observation(observation)
        self.validation_observation = deepcopy(observation)
        self.memory.restore(record.get('memory'), resolve_lookup)
        requests, history = record.get('requests'), record.get('history')
        if (not isinstance(requests, list) or not isinstance(history, list)
                or len(history) > PAGE_SIZE
                or any(not isinstance(row, list) or len(row) != 3 for row in requests)
                or any(not isinstance(row, dict) for row in history)):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        restored = {_lookup_request(*row) for row in requests}
        if len(restored) != len(requests):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        cursors = record.get('lookup_cursors')
        if (not isinstance(cursors, list) or any(
                not isinstance(row, dict) or set(row) != {'kind', 'query', 'offset'}
                or _lookup_request(row['kind'], row['query'], row['offset']) not in restored
                for row in cursors)):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        cursor_requests = [_lookup_request(row['kind'], row['query'], row['offset']) for row in cursors]
        latest = {}
        for kind, query, offset in restored:
            latest[(kind, query)] = max(offset, latest.get((kind, query), -1))
        if {(kind, query): offset for kind, query, offset in cursor_requests} != latest or len(cursors) != len(latest):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        if cursors and not callable(resolve_lookup):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_LOOKUP_RESOLVER_REQUIRED')
        last = record.get('last_lookup')
        if last is not None and (not isinstance(last, dict) or set(last) != {'kind', 'query', 'offset'}
                or _lookup_request(last['kind'], last['query'], last['offset']) not in restored):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        reads = record.get('operation_reads', len(restored) if self.phase == 'operation_graph' else 0)
        if type(reads) is not int or reads < 0:
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        self.operation_reads = reads
        self.requests = restored
        self.history = deque(deepcopy(history), maxlen=PAGE_SIZE)
        self.lookup_cursors = {}
        for row in cursors:
            result = resolve_lookup(row['kind'], row['query'], row['offset'])
            self._remember_cursor(row['kind'], row['query'], row['offset'], result)
        # The caller re-resolves this selector against current inspected inputs.
        self.lookup = {}
        self.symbol_lookup = {}
        binding = record.get('last_binding_lookup')
        if binding is not None:
            if (not isinstance(binding, dict) or set(binding) != {'kind', 'query', 'offset'}
                    or binding['kind'] != 'bindings'):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
            _lookup_request(binding['kind'], binding['query'], binding['offset'])
            self.binding_lookup = {**self.binding_result(binding['query'], binding['offset']), **binding}
        else:
            self.binding_lookup = {}
        binding_views = record.get('binding_lookups')
        if not isinstance(binding_views, list):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        self.binding_lookups = {}
        for view in binding_views:
            if (not isinstance(view, dict) or set(view) != {'kind', 'query', 'offset'}
                    or view['kind'] != 'bindings'):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
            _lookup_request(view['kind'], view['query'], view['offset'])
            key = (view['query'], view['offset'])
            if key in self.binding_lookups:
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
            self.binding_lookups[key] = {**self.binding_result(view['query'], view['offset']), **view}
        if binding is not None and (binding['query'], binding['offset']) not in self.binding_lookups:
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        symbol = record.get('last_symbol_lookup')
        if symbol is not None:
            if (not isinstance(symbol, dict) or set(symbol) != {'kind', 'query', 'offset'}
                    or symbol['kind'] != 'symbols'):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
            _lookup_request(symbol['kind'], symbol['query'], symbol['offset'])
            self.symbol_lookup = {**self.symbol_result(symbol['query'], symbol['offset']), **symbol}
        source = record.get('last_source_lookup')
        self.source_lookup = {}
        if source is not None:
            if (not isinstance(source, dict) or set(source) != {'kind', 'query', 'offset'}
                    or source['kind'] != 'source' or not callable(resolve_lookup)):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
            _lookup_request(source['kind'], source['query'], source['offset'])
            self.source_lookup = {**resolve_lookup(source['kind'], source['query'], source['offset']), **source}
        scope = record.get('last_scope_lookup')
        self.scope_lookup = {}
        if scope is not None:
            if (not isinstance(scope, dict) or set(scope) != {'kind', 'query', 'offset'}
                    or scope['kind'] != 'scopes' or self.evidence_index is None):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
            _lookup_request(scope['kind'], scope['query'], scope['offset'])
            self.scope_lookup = {**self.evidence_index.scope_page(scope['query'], scope['offset']), **scope}
        return last

    def symbol_result(self, query, offset):
        if self.evidence_index is not None:
            return self.evidence_index.page(query, offset)
        kinds = {'CLASS'} if self.phase in TYPE_LOOKUP_PHASES else None
        return symbol_page(self.symbols, query, offset, kinds=kinds)

    def lookup_symbols(self, query, offset):
        return self.consume_lookup("symbols", query, offset,
                                   self.symbol_result(query, offset))

    def lookup_scopes(self, query, offset):
        if self.evidence_index is None:
            raise ValueError('CONSTRUCTION_RETRIEVAL_GOAL_MISSING')
        result = self.consume_lookup('scopes', query, offset, self.evidence_index.scope_page(query, offset))
        self.symbol_goal = query
        if ('symbols', symbol_query_key(query), 0) not in self.requests:
            self.lookup_symbols(query, 0)
        else:
            cursor = self.lookup_cursors[('symbols', symbol_query_key(query))]
            self.symbol_lookup = self.symbol_result(query, cursor['offset'])
        return result

    def next_declaration_offset(self):
        # The schema and reader share this cursor. Every page remains reachable,
        # but an already consumed page cannot be offered as a fresh decision.
        for offset in range(0, max(1, len(self.declaration_ids)), PAGE_SIZE):
            if ('declarations', '', offset) not in self.requests:
                return offset
        return None

    def lookup_declarations(self, offset):
        rows = [symbol_record(key, self.symbols[key]) for key in self.declaration_ids]
        return self.consume_lookup("declarations", "", offset,
                                   page_result(rows, offset, field="declarations"))

    def binding_result(self, query, offset):
        if query not in self.binding_pages:
            raise ValueError('CONSTRUCTION_BINDING_MISSING')
        return page_result(self.binding_pages[query], offset, field='bindings')

    def lookup_bindings(self, query, offset):
        return self.consume_lookup('bindings', query, offset, self.binding_result(query, offset))
